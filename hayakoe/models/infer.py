from typing import Any, Optional

import torch
from numpy.typing import NDArray

from hayakoe.constants import Languages
from hayakoe.logging import logger
from hayakoe.models import utils
from hayakoe.models.boundary_pauses import (
    durations_to_boundary_pauses,
    find_boundary_punct_positions,
)
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from hayakoe.models.text_preprocess import prepare_phone_sequences
from hayakoe.nlp import extract_bert_feature
from hayakoe.nlp.symbols import SYMBOLS


def get_net_g(
    model_path: str, version: str, device: str, hps: HyperParameters
) -> SynthesizerTrnJPExtra:
    if not version.endswith("JP-Extra"):
        raise ValueError(f"Only JP-Extra models are supported, got version: {version}")

    logger.info("Using JP-Extra model")
    net_g = SynthesizerTrnJPExtra(
        n_vocab=len(SYMBOLS),
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=hps.model.use_duration_discriminator,
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
    ).to(device)
    _ = net_g.eval()
    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g, None, skip_optimizer=True, device=device
        )
    elif model_path.endswith(".safetensors"):
        _ = utils.safetensors.load_safetensors(model_path, net_g, True, device=device)
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    return net_g


def get_text(
    text: str,
    hps: HyperParameters,
    device: str,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_text, phone, tone, language, word2ph = prepare_phone_sequences(
        text,
        hps,
        given_phone=given_phone,
        given_tone=given_tone,
    )

    ja_bert = extract_bert_feature(
        norm_text,
        word2ph,
        Languages.JP,
        device,
    )
    assert ja_bert.shape[-1] == len(phone), phone

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return ja_bert, phone, tone, language


def infer(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,
    hps: HyperParameters,
    net_g: SynthesizerTrnJPExtra,
    device: str,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """PyTorch 추론: 텍스트 → 오디오.

    Returns:
        ``(audio, phone_ids, durations)``. ``durations`` 는 phoneme 별 frame 수로,
        합성에 실제로 쓰인 attention path 에서 복원하므로 오디오와 정확히 맞는다.
    """
    ja_bert, phones, tones, lang_ids = get_text(
        text,
        hps,
        device,
        given_phone=given_phone,
        given_tone=given_tone,
    )

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        sid_tensor = torch.LongTensor([sid]).to(device)

        output = net_g.infer(
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones,
            lang_ids,
            ja_bert,
            style_vec=style_vec_tensor,
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
        )

        audio = output[0][0, 0].data.cpu().float().numpy()
        # attn 은 [b, 1, frame, phoneme] path — frame 축으로 더하면 합성에 쓰인
        # w_ceil (phoneme 별 frame 수) 이 그대로 복원된다.
        durations = output[1].sum(2)[0, 0].data.cpu().float().numpy()
        phone_ids = phones.data.cpu().numpy()

        del (
            x_tst,
            tones,
            lang_ids,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            style_vec,
        )
        return audio, phone_ids, durations


def predict_boundary_pauses(
    text: str,
    style_vec: NDArray[Any],
    length_scale: float,
    sid: int,
    num_sentences: int,
    hps: HyperParameters,
    net_g: SynthesizerTrnJPExtra,
    device: str,
    sdp_ratio: float = 0.0,
    noise_scale_w: float = 0.8,
) -> list[float]:
    """전체 텍스트에서 문장 경계의 pause 길이(초)를 예측한다 (PyTorch).

    Text encoder + duration predictor만 실행하므로 decoder 대비 매우 가볍다.

    Returns:
        문장 경계별 pause 길이 리스트 (len = num_sentences - 1).
    """
    ja_bert, phones, tones, lang_ids = get_text(text, hps, device)

    phone_list = phones.tolist()
    punct_positions = find_boundary_punct_positions(phone_list)
    if not punct_positions or num_sentences <= 1:
        return []

    with torch.no_grad():
        durations = net_g.predict_durations(
            phones.to(device).unsqueeze(0),
            torch.LongTensor([phones.size(0)]).to(device),
            torch.LongTensor([sid]).to(device),
            tones.to(device).unsqueeze(0),
            lang_ids.to(device).unsqueeze(0),
            ja_bert.to(device).unsqueeze(0),
            torch.from_numpy(style_vec).to(device).unsqueeze(0),
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
            noise_scale_w=noise_scale_w,
        ).cpu().numpy()

    return durations_to_boundary_pauses(
        durations, phone_list, punct_positions, num_sentences, hps,
    )
