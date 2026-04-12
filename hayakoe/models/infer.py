from typing import Any, Optional

import torch
from numpy.typing import NDArray

from hayakoe.constants import Languages
from hayakoe.logging import logger
from hayakoe.models import commons, utils
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)

SynthesizerTrn = SynthesizerTrnJPExtra
from hayakoe.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from hayakoe.nlp.symbols import SYMBOLS

_BOUNDARY_PUNCT_IDS = frozenset(
    SYMBOLS.index(p) for p in (".", "!", "?") if p in SYMBOLS
)


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
    net_g.state_dict()
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
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
        text,
        Languages.JP,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, Languages.JP)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    ja_bert = extract_bert_feature(
        norm_text,
        word2ph,
        Languages.JP,
        device,
        assist_text,
        assist_text_weight,
    )
    del word2ph
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
    language: Languages,
    hps: HyperParameters,
    net_g: SynthesizerTrnJPExtra,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> NDArray[Any]:
    ja_bert, phones, tones, lang_ids = get_text(
        text,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        ja_bert = ja_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        ja_bert = ja_bert[:, :-2]

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
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

        del (
            x_tst,
            tones,
            lang_ids,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            style_vec,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio


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
    """전체 텍스트에서 문장 경계의 pause 길이(초)를 예측한다.

    Text encoder + duration predictor만 실행하므로 decoder 대비 매우 가볍다.

    Returns:
        문장 경계별 pause 길이 리스트 (len = num_sentences - 1).
    """
    ja_bert, phones, tones, lang_ids = get_text(text, hps, device)

    phone_list = phones.tolist()
    punct_positions = [i for i, p in enumerate(phone_list) if p in _BOUNDARY_PUNCT_IDS]

    num_boundaries = num_sentences - 1
    if not punct_positions or num_boundaries <= 0:
        return []

    hop_length = hps.data.hop_length
    sr = hps.data.sampling_rate

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

    pauses: list[float] = []
    for pos in punct_positions[:num_boundaries]:
        frames = float(durations[pos])
        if pos > 0 and phone_list[pos - 1] == 0:
            frames += float(durations[pos - 1])
        if pos + 1 < len(phone_list) and phone_list[pos + 1] == 0:
            frames += float(durations[pos + 1])
        pauses.append(frames * hop_length / sr)

    return pauses
