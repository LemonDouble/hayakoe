from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from hayakoe.api.audio_result import AudioResult, StyleAccessor
from hayakoe.constants import Languages
from hayakoe.logging import logger
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.voice import adjust_voice


if TYPE_CHECKING:
    from hayakoe.models.models_jp_extra import (
        SynthesizerTrn as SynthesizerTrnJPExtra,
    )


class Speaker:
    """로드된 화자 모델. TTS.load()를 통해 생성된다."""

    def __init__(
        self,
        name: str,
        model_dir: Path,
        device: str,
        backend: str = "onnx",
        bert_session=None,
    ) -> None:
        self.name = name
        self._device = device
        self._backend = backend
        self._model_dir = model_dir
        self._bert_session = bert_session

        self._config_path = model_dir / "config.json"
        self._style_vec_path = model_dir / "style_vectors.npy"

        if not self._config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        if not self._style_vec_path.exists():
            raise FileNotFoundError(f"style_vectors.npy not found in {model_dir}")

        # 설정 + 스타일 벡터 로드
        self._hps = HyperParameters.load_from_json(self._config_path)
        self._style_vectors: NDArray = np.load(self._style_vec_path)

        if hasattr(self._hps.data, "style2id"):
            self._style2id: dict[str, int] = self._hps.data.style2id
        else:
            num_styles = self._hps.data.num_styles
            self._style2id = {str(i): i for i in range(num_styles)}

        self.styles = StyleAccessor(self._style2id)

        # 지연 로드 (백엔드별)
        self._net_g: Optional[SynthesizerTrnJPExtra] = None
        self._synth_session = None

        if backend == "onnx":
            self._init_onnx_synth()

        logger.info(
            f"Speaker '{name}' loaded ({backend}, "
            f"styles: {list(self._style2id.keys())})"
        )

    def _init_onnx_synth(self):
        """ONNX Synthesizer 세션을 생성한다."""
        import onnxruntime as ort

        # synthesizer.onnx 우선, 없으면 synthesizer_q8.onnx
        onnx_path = self._model_dir / "synthesizer.onnx"
        if not onnx_path.exists():
            onnx_path = self._model_dir / "synthesizer_q8.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"No synthesizer.onnx in {self._model_dir}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._synth_session = ort.InferenceSession(
            str(onnx_path), sess_opts, providers=["CPUExecutionProvider"]
        )

    def _ensure_pytorch_model(self) -> SynthesizerTrnJPExtra:
        """PyTorch 모델을 지연 로드한다."""
        if self._net_g is not None:
            return self._net_g

        from hayakoe.models.infer import get_net_g

        safetensors_files = sorted(
            self._model_dir.glob("*.safetensors"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors in {self._model_dir}")

        self._net_g = get_net_g(
            model_path=str(safetensors_files[0]),
            version=self._hps.version,
            device=self._device,
            hps=self._hps,
        )
        return self._net_g

    def _get_style_vector(self, style: str, weight: float) -> NDArray:
        style_id = self._style2id.get(style)
        if style_id is None:
            available = list(self._style2id.keys())
            raise ValueError(f"Style '{style}' not found. Available: {available}")
        mean = self._style_vectors[0]
        vec = self._style_vectors[style_id]
        return mean + (vec - mean) * weight

    def generate(
        self,
        text: str,
        *,
        lang: Union[str, Languages] = Languages.JP,
        style: str = "Neutral",
        speaker_id: int = 0,
        speed: float = 1.0,
        sdp_ratio: float = 0.2,
        noise: float = 0.6,
        noise_w: float = 0.8,
        pitch_scale: float = 1.0,
        intonation_scale: float = 1.0,
        style_weight: float = 1.0,
    ) -> AudioResult:
        """텍스트에서 음성을 생성한다.

        Args:
            text: 합성할 텍스트.
            lang: 언어 (Lang.JA).
            style: 스타일 이름 (예: "Neutral", "Happy").
            speaker_id: 멀티 화자 모델용 화자 ID.
            speed: 말속도. 1.0 = 보통, <1.0 = 빠름, >1.0 = 느림.
            sdp_ratio: SDP/DP 비율.
            noise: DP용 노이즈 스케일.
            noise_w: SDP용 노이즈 스케일.
            pitch_scale: 피치 조정 (1.0 = 변경 없음).
            intonation_scale: 억양 조정 (1.0 = 변경 없음).
            style_weight: 스타일 벡터 가중치.

        Returns:
            .save()와 .to_bytes() 메서드를 가진 AudioResult.
        """
        lang_str = Languages(lang.value if hasattr(lang, "value") else str(lang))
        style_vec = self._get_style_vector(style, style_weight)

        if self._backend == "onnx":
            audio = self._generate_onnx(
                text, lang_str, style_vec, speaker_id,
                speed, sdp_ratio, noise, noise_w,
            )
        else:
            audio = self._generate_pytorch(
                text, lang_str, style_vec, speaker_id,
                speed, sdp_ratio, noise, noise_w,
            )

        if pitch_scale != 1.0 or intonation_scale != 1.0:
            _, audio = adjust_voice(
                fs=self._hps.data.sampling_rate,
                wave=audio,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )

        # 16-bit PCM 변환
        audio = audio / np.abs(audio).max()
        audio = (audio * 32767).astype(np.int16)

        return AudioResult(sr=self._hps.data.sampling_rate, data=audio)

    def _generate_onnx(self, text, lang, style_vec, sid, speed, sdp_ratio, noise, noise_w):
        from hayakoe.models.infer_onnx import infer_onnx

        return infer_onnx(
            text=text,
            style_vec=style_vec,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noise_w,
            length_scale=speed,
            sid=sid,
            language=lang,
            hps=self._hps,
            bert_session=self._bert_session,
            synth_session=self._synth_session,
        )

    def _generate_pytorch(self, text, lang, style_vec, sid, speed, sdp_ratio, noise, noise_w):
        import torch

        from hayakoe.models.infer import infer

        net_g = self._ensure_pytorch_model()
        with torch.no_grad():
            return infer(
                text=text,
                sdp_ratio=sdp_ratio,
                noise_scale=noise,
                noise_scale_w=noise_w,
                length_scale=speed,
                sid=sid,
                language=lang,
                hps=self._hps,
                net_g=net_g,
                device=self._device,
                style_vec=style_vec,
            )

    def __repr__(self) -> str:
        return f"Speaker('{self.name}', backend='{self._backend}', styles={list(self._style2id.keys())})"
