"""JP-Extra 모델 학습 스크립트 (분산 학습 지원)."""

import argparse
import datetime
import gc
import os
import warnings

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

import default_style
from config import get_config
from data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from losses import WavLMLoss, discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from hayakoe.logging import logger
from hayakoe.models import commons, utils
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.models.models_jp_extra import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    WavLMDiscriminator,
)
from hayakoe.nlp.symbols import SYMBOLS
from hayakoe.utils.stdout_wrapper import SAFE_STDOUT


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("medium")
torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION)

config = get_config()
global_step = 0


def _format_eta(seconds: int) -> str:
    """초를 HH:MM:SS 또는 MM:SS 형식으로 변환."""
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str,
        default=config.train_ms_config.config_path,
        help="학습 설정 JSON 파일 경로",
    )
    parser.add_argument(
        "-m", "--model", type=str,
        default=config.dataset_path,
        help="데이터셋 폴더 경로",
    )
    parser.add_argument(
        "--assets_root", type=str,
        default=config.assets_root,
        help="추론용 모델 에셋 루트 디렉토리",
    )
    parser.add_argument(
        "--skip_default_style", action="store_true",
        help="기본 스타일 벡터 저장 스킵",
    )
    parser.add_argument(
        "--wav_dir", type=str, default=None,
        help="기본 스타일 벡터용 wav 디렉토리",
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true",
        help="프로그레스바 비활성화",
    )
    parser.add_argument(
        "--speedup", action="store_true",
        help="로깅/평가 비활성화로 학습 속도 향상",
    )
    parser.add_argument(
        "--not_use_custom_batch_sampler", action="store_true",
        help="커스텀 배치 샘플러 대신 DistributedLengthGroupedSampler 사용",
    )
    args = parser.parse_args()

    # 로그 파일 설정
    model_dir = os.path.join(args.model, config.train_ms_config.model_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(os.path.join(args.model, f"train_{timestamp}.log"))

    # 환경변수 로드
    envs = config.train_ms_config.env
    for env_name, env_value in envs.items():
        if env_name not in os.environ.keys():
            logger.info(f"설정에서 환경변수 로드: {env_value!s}")
            os.environ[env_name] = str(env_value)
    logger.info(
        "환경변수: MASTER_ADDR={}, MASTER_PORT={}, WORLD_SIZE={}, RANK={}, LOCAL_RANK={}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = dist.get_world_size()

    hps = HyperParameters.load_from_json(args.config)
    hps.model_dir = model_dir
    hps.speedup = args.speedup

    # 설정 파일 경로가 다르면 복사
    if os.path.realpath(args.config) != os.path.realpath(
        config.train_ms_config.config_path
    ):
        with open(args.config, encoding="utf-8") as f:
            data = f.read()
        os.makedirs(os.path.dirname(config.train_ms_config.config_path), exist_ok=True)
        with open(config.train_ms_config.config_path, "w", encoding="utf-8") as f:
            f.write(data)

    os.makedirs(config.out_dir, exist_ok=True)

    if not args.skip_default_style:
        wav_dir = args.wav_dir or os.path.join(args.model, "audio")
        default_style.save_styles_by_dirs(
            wav_dir,
            config.out_dir,
            config_path=args.config,
            config_output_path=os.path.join(config.out_dir, "config.json"),
        )

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(local_rank)

    global global_step
    writer = None
    writer_eval = None
    if rank == 0 and not args.speedup:
        writer = SummaryWriter(log_dir=model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    if not args.not_use_custom_batch_sampler:
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
        )
    else:
        train_sampler = DistributedLengthGroupedSampler(
            dataset=train_dataset,
            batch_size=hps.train.batch_size,
            num_replicas=n_gpus,
            rank=rank,
            lengths=train_dataset.lengths,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            batch_size=hps.train.batch_size,
            persistent_workers=True,
        )
        logger.info("DistributedLengthGroupedSampler 사용 중")
        logger.debug(f"len(train_dataset): {len(train_dataset)}")
        logger.debug(f"len(train_loader): {len(train_loader)}")

    eval_dataset = None
    eval_loader = None
    if rank == 0 and not args.speedup:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    # Noise Scaled MAS (VITS2)
    if hps.model.use_noise_scaled_mas is True:
        logger.info("VITS2 Noise Scaled MAS 사용")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    # WavLM 판별기
    if hps.model.use_wavlm_discriminator is True:
        net_wd = WavLMDiscriminator(
            hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
        ).cuda(local_rank)
    else:
        net_wd = None

    if hps.model.use_spk_conditioned_encoder is True:
        if hps.data.n_speakers == 0:
            raise ValueError(
                "화자 조건부 인코더 사용 시 n_speakers > 0 이어야 합니다"
            )

    net_g = SynthesizerTrn(
        len(SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=False,
        use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
        slm=hps.model.slm,
    ).cuda(local_rank)

    if getattr(hps.train, "freeze_JP_bert", False):
        logger.info("JP BERT 인코더 동결")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False
    if getattr(hps.train, "freeze_style", False):
        logger.info("스타일 인코더 동결")
        for param in net_g.enc_p.style_proj.parameters():
            param.requires_grad = False
    if getattr(hps.train, "freeze_decoder", False):
        logger.info("디코더 동결")
        for param in net_g.dec.parameters():
            param.requires_grad = False

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(local_rank)
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_wd is not None:
        optim_wd = torch.optim.AdamW(
            net_wd.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_wd = None

    # DDP bucket view stride mismatch 경고 suppress (단일 GPU에서 무해)
    warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

    net_g = DDP(net_g, device_ids=[local_rank], gradient_as_bucket_view=False)
    net_d = DDP(net_d, device_ids=[local_rank], gradient_as_bucket_view=False)
    if net_wd is not None:
        net_wd = DDP(net_wd, device_ids=[local_rank], gradient_as_bucket_view=False)

    if utils.is_resuming(model_dir):
        # WavLM 판별기 체크포인트 복원
        if net_wd is not None:
            try:
                _, optim_wd, wd_resume_lr, epoch_str = (
                    utils.checkpoints.load_checkpoint(
                        utils.checkpoints.get_latest_checkpoint_path(
                            model_dir, "WD_*.pth"
                        ),
                        net_wd,
                        optim_wd,
                        skip_optimizer=hps.train.skip_optimizer,
                    )
                )
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
            except Exception:
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
                logger.info("WavLM 판별기 초기화")

        # Generator / Discriminator 체크포인트 복원
        try:
            _, optim_g, g_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr

            epoch_str = max(epoch_str, 1)
            global_step = int(
                utils.get_steps(
                    utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth")
                )
            )
            logger.info(
                f"체크포인트 발견. 현재 에포크: {epoch_str}, 글로벌 스텝: {global_step}"
            )
        except Exception as e:
            logger.warning(e)
            logger.warning("사전학습 모델을 찾을 수 없어 처음부터 학습합니다.")
            epoch_str = 1
            global_step = 0
    else:
        # safetensors 프리트레인 모델 로드 시도
        try:
            _ = utils.safetensors.load_safetensors(
                os.path.join(model_dir, "G_0.safetensors"),
                net_g,
                expected_missing_keys=[
                    "enc_p.style_proj.weight",
                    "enc_p.style_proj.bias",
                    "emb_g.weight",
                ],
            )
            _ = utils.safetensors.load_safetensors(
                os.path.join(model_dir, "D_0.safetensors"), net_d
            )
            if net_wd is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(model_dir, "WD_0.safetensors"), net_wd
                )
            logger.info("사전학습 모델 로드 완료.")
        except Exception as e:
            logger.warning(e)
            logger.warning("사전학습 모델을 찾을 수 없어 처음부터 학습합니다.")
        finally:
            epoch_str = 1
            global_step = 0

    def lr_lambda(epoch):
        """워밍업 + 지수 감쇠 학습률 스케줄러."""
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        else:
            return hps.train.lr_decay ** (epoch - hps.train.warmup_epochs)

    scheduler_last_epoch = epoch_str - 2
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    if net_wd is not None:
        scheduler_wd = torch.optim.lr_scheduler.LambdaLR(
            optim_wd, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
        wl = WavLMLoss(
            hps.model.slm.model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ).to(local_rank)
    else:
        scheduler_wd = None
        wl = None
    scaler = GradScaler("cuda", enabled=hps.train.bf16_run)
    logger.info("학습 시작.")

    diff = abs(
        epoch_str * len(train_loader) - (hps.train.epochs + 1) * len(train_loader)
    )
    pbar = None
    if not args.no_progress_bar:
        pbar = tqdm(
            total=global_step + diff,
            initial=global_step,
            smoothing=0.05,
            file=SAFE_STDOUT,
            dynamic_ncols=True,
            unit="step",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}|{rate_fmt}]",
        )
    initial_step = global_step

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank, local_rank, epoch, hps,
                [net_g, net_d, net_wd, wl],
                [optim_g, optim_d, optim_wd],
                [scheduler_g, scheduler_d, scheduler_wd],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
                pbar, initial_step,
            )
        else:
            train_and_evaluate(
                rank, local_rank, epoch, hps,
                [net_g, net_d, net_wd, wl],
                [optim_g, optim_d, optim_wd],
                [scheduler_g, scheduler_d, scheduler_wd],
                scaler,
                [train_loader, None],
                None, None,
                pbar, initial_step,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_wd is not None:
            scheduler_wd.step()

        # 마지막 에포크: 최종 모델 저장
        if epoch == hps.train.epochs:
            utils.checkpoints.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(model_dir, f"G_{global_step}.pth"),
            )
            utils.checkpoints.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(model_dir, f"D_{global_step}.pth"),
            )
            if net_wd is not None:
                utils.checkpoints.save_checkpoint(
                    net_wd, optim_wd, hps.train.learning_rate, epoch,
                    os.path.join(model_dir, f"WD_{global_step}.pth"),
                )
            utils.safetensors.save_safetensors(
                net_g, epoch,
                os.path.join(
                    config.out_dir,
                    f"{config.model_name}_e{epoch}_s{global_step}.safetensors",
                ),
                for_infer=True,
            )

    if pbar is not None:
        pbar.close()


def train_and_evaluate(
    rank, local_rank, epoch, hps,
    nets, optims, schedulers, scaler,
    loaders, logger, writers,
    pbar: tqdm, initial_step: int,
):
    net_g, net_d, net_wd, wl = nets
    optim_g, optim_d, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_wd = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    global global_step

    net_g.train()
    net_d.train()
    if net_wd is not None:
        net_wd.train()

    for batch_idx, (
        x, x_lengths, spec, spec_lengths,
        y, y_lengths, speakers,
        tone, language, bert, style_vec,
    ) in enumerate(train_loader):
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        x, x_lengths = x.cuda(local_rank, non_blocking=True), x_lengths.cuda(local_rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(local_rank, non_blocking=True), spec_lengths.cuda(local_rank, non_blocking=True)
        y, y_lengths = y.cuda(local_rank, non_blocking=True), y_lengths.cuda(local_rank, non_blocking=True)
        speakers = speakers.cuda(local_rank, non_blocking=True)
        tone = tone.cuda(local_rank, non_blocking=True)
        language = language.cuda(local_rank, non_blocking=True)
        bert = bert.cuda(local_rank, non_blocking=True)
        style_vec = style_vec.cuda(local_rank, non_blocking=True)

        with autocast("cuda", enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            (
                y_hat, l_length, attn, ids_slice,
                x_mask, z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                g,
            ) = net_g(
                x, x_lengths, spec, spec_lengths,
                speakers, tone, language, bert, style_vec,
            )
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

            # 판별기 학습
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast("cuda", enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

            if net_wd is not None:
                with autocast("cuda", enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                    loss_slm = wl.discriminator(
                        y.detach().squeeze(1), y_hat.detach().squeeze(1)
                    ).mean()

                optim_wd.zero_grad()
                scaler.scale(loss_slm).backward()
                scaler.unscale_(optim_wd)
                grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
                scaler.step(optim_wd)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast("cuda", enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            # 생성기 학습
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_wd is not None:
                loss_lm = wl(y.detach().squeeze(1), y_hat.squeeze(1)).mean()
                loss_lm_gen = wl.generator(y_hat.squeeze(1))
            with autocast("cuda", enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_wd is not None:
                    loss_gen_all += loss_lm + loss_lm_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            # 텐서보드 로깅
            if global_step % hps.train.log_interval == 0 and not hps.speedup:
                lr = optim_g.param_groups[0]["lr"]
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/dur": loss_dur,
                    "loss/g/kl": loss_kl,
                }
                scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
                scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})

                if net_wd is not None:
                    scalar_dict.update({
                        "loss/wd/total": loss_slm,
                        "grad_norm_wd": grad_norm_wd,
                        "loss/g/lm": loss_lm,
                        "loss/g/lm_gen": loss_lm_gen,
                    })
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

            # 체크포인트 저장 (eval_interval마다)
            if (
                global_step % hps.train.eval_interval == 0
                and global_step != 0
                and initial_step != global_step
            ):
                if not hps.speedup:
                    evaluate(hps, net_g, eval_loader, writer_eval)
                utils.checkpoints.save_checkpoint(
                    net_g, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"G_{global_step}.pth"),
                )
                utils.checkpoints.save_checkpoint(
                    net_d, optim_d, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"D_{global_step}.pth"),
                )
                if net_wd is not None:
                    utils.checkpoints.save_checkpoint(
                        net_wd, optim_wd, hps.train.learning_rate, epoch,
                        os.path.join(hps.model_dir, f"WD_{global_step}.pth"),
                    )
                keep_ckpts = config.train_ms_config.keep_ckpts
                if keep_ckpts > 0:
                    utils.checkpoints.clean_checkpoints(
                        model_dir_path=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )
                # 추론용 safetensors 저장
                utils.safetensors.save_safetensors(
                    net_g, epoch,
                    os.path.join(
                        config.out_dir,
                        f"{config.model_name}_e{epoch}_s{global_step}.safetensors",
                    ),
                    for_infer=True,
                )

        global_step += 1
        if pbar is not None:
            elapsed = pbar.format_dict["elapsed"]
            steps_done = pbar.n + 1 - pbar.initial
            if steps_done > 0 and elapsed > 0:
                sec_per_step = elapsed / steps_done
                # 다음 저장까지 남은 step
                interval = hps.train.eval_interval
                steps_to_save = interval - (global_step % interval)
                if steps_to_save == interval:
                    steps_to_save = 0
                # 전체 완료까지 남은 step
                steps_to_end = pbar.total - (pbar.n + 1)
                eta_save = _format_eta(int(steps_to_save * sec_per_step))
                eta_end = _format_eta(int(steps_to_end * sec_per_step))
                pbar.bar_format = (
                    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}|{rate_fmt}|"
                    f"다음 checkpoint 저장 {eta_save} | 학습 완료까지 {eta_end}]"
                )
            pbar.set_description(
                f"에포크 {epoch}/{hps.train.epochs}"
            )
            pbar.update()

    gc.collect()
    torch.cuda.empty_cache()
    if pbar is None and rank == 0:
        logger.info(f"====> 에포크: {epoch}, 스텝: {global_step}")


def evaluate(hps, generator, eval_loader, writer_eval):
    """검증 데이터셋으로 모델 평가 후 텐서보드에 오디오 기록."""
    generator.eval()
    audio_dict = {}
    logger.info("평가 중...")
    with torch.no_grad():
        for batch_idx, (
            x, x_lengths, spec, spec_lengths,
            y, y_lengths, speakers,
            tone, language, bert, style_vec,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            style_vec = style_vec.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x, x_lengths, speakers, tone, language,
                    bert, style_vec,
                    y=spec, max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                audio_dict.update({
                    f"gen/audio_{batch_idx}_{use_sdp}": y_hat[0, :, : y_hat_lengths[0]]
                })
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images={},
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    run()
