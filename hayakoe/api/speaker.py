from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from hayakoe.api.audio_result import AudioResult, StyleAccessor
from hayakoe.constants import Languages
from hayakoe.logging import logger
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.voice import adjust_voice

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])")


if TYPE_CHECKING:
    from hayakoe.models.models_jp_extra import (
        SynthesizerTrn as SynthesizerTrnJPExtra,
    )


class Speaker:
    """로드된 화자 모델. :meth:`TTS.load` 를 통해 생성된다.

    백엔드별 동작:

    - **onnx** (CPU): ONNX Runtime으로 추론. ``TTS(device="cpu")`` 사용 시 자동 선택.
    - **pytorch** (CUDA): PyTorch eager mode. ``TTS(device="cuda")`` 사용 시 자동 선택.
    - **compiled** (CUDA): ``tts.optimize()`` 호출 후 torch.compile 적용. 10-25% 향상.

    사용법::

        from hayakoe import TTS

        # CPU
        speaker = TTS().load("jvnv-F1-jp")
        speaker.generate("こんにちは").save("output.wav")

        # GPU + torch.compile
        tts = TTS(device="cuda")
        tts.load("jvnv-F1-jp")
        tts.optimize()  # 로드된 전체 화자에 torch.compile 적용
        tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
    """

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

        여러 문장이 포함된 텍스트는 문장 경계(。！？!?\\n)에서
        자동 분할하여 개별 추론 후 연결한다.

        Args:
            text: 합성할 텍스트.
            lang: 언어. 현재 일본어(``Languages.JP``)만 지원.
            style: 감정 스타일. ``"Neutral"``, ``"Happy"``, ``"Sad"``,
                ``"Angry"``, ``"Fear"``, ``"Surprise"``, ``"Disgust"``.
            speaker_id: 멀티 화자 모델용 화자 ID.
            speed: 말속도. 1.0 = 보통, <1.0 = 빠름, >1.0 = 느림.
            sdp_ratio: SDP/DP 비율 (0.0-1.0). 높을수록 억양 변화 큼.
            noise: 음성 변동성 (0.0-1.0).
            noise_w: 발화 리듬 변동성 (0.0-1.0).
            pitch_scale: 피치 배율 (1.0 = 변경 없음).
            intonation_scale: 억양 배율 (1.0 = 변경 없음).
            style_weight: 스타일 벡터 가중치 (0.0-1.0).

        Returns:
            ``.save(path)`` 와 ``.to_bytes()`` 메서드를 가진
            :class:`AudioResult`.

        Example::

            audio = speaker.generate(
                "今日はどんな国に辿り着くのでしょうか。",
                style="Happy",
                speed=0.9,
            )
            audio.save("output.wav")
        """
        kwargs = dict(
            lang=lang, style=style, speaker_id=speaker_id,
            speed=speed, sdp_ratio=sdp_ratio, noise=noise, noise_w=noise_w,
            pitch_scale=pitch_scale, intonation_scale=intonation_scale,
            style_weight=style_weight,
        )

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            audio = self._synthesize_one(text, **kwargs)
            return self._to_audio_result(audio)

        sr = self._hps.data.sampling_rate
        silence = np.zeros(int(sr * 0.2), dtype=np.float32)  # 200ms

        parts = []
        for i, sentence in enumerate(sentences):
            if i > 0:
                parts.append(silence)
            parts.append(self._synthesize_one(sentence, **kwargs))

        return self._to_audio_result(np.concatenate(parts))

    def stream(
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
        silence_ms: int = 200,
    ) -> Generator[AudioResult, None, None]:
        """텍스트를 문장 단위로 스트리밍 생성한다.

        문장 경계(。！？!?\\n)에서 분할하여 각 문장을 개별 추론하고,
        완료된 순서대로 yield한다. 첫 문장이 완성되는 즉시
        재생을 시작할 수 있어 체감 지연이 줄어든다.

        Args:
            text: 합성할 텍스트.
            silence_ms: 문장 사이에 삽입할 무음 길이 (밀리초, 기본 200).
            **kwargs: :meth:`generate` 와 동일한 파라미터.

        Yields:
            문장별 :class:`AudioResult`. 두 번째 문장부터 앞에 무음이 포함된다.

        Example::

            for chunk in speaker.stream("こんにちは。元気ですか？"):
                play(chunk.to_bytes())  # 문장별로 바로 재생
        """
        sentences = _split_sentences(text)
        if not sentences:
            return

        sr = self._hps.data.sampling_rate
        silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.int16)

        for i, sentence in enumerate(sentences):
            audio = self._synthesize_one(
                sentence, lang=lang, style=style, speaker_id=speaker_id,
                speed=speed, sdp_ratio=sdp_ratio, noise=noise, noise_w=noise_w,
                pitch_scale=pitch_scale, intonation_scale=intonation_scale,
                style_weight=style_weight,
            )
            pcm = self._to_pcm(audio)

            if i > 0:
                pcm = np.concatenate([silence, pcm])

            yield AudioResult(sr=sr, data=pcm)

    def _synthesize_one(
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
    ) -> NDArray:
        """단일 텍스트 → float32 오디오 배열."""
        lang_str = Languages(lang.value if hasattr(lang, "value") else str(lang))
        style_vec = self._get_style_vector(style, style_weight)

        if self._backend == "onnx":
            audio = self._generate_onnx(
                text, lang_str, style_vec, speaker_id,
                speed, sdp_ratio, noise, noise_w,
            )
        else:
            # pytorch / compiled 모두 같은 경로
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

        return audio

    @staticmethod
    def _to_pcm(audio: NDArray) -> NDArray[np.int16]:
        """float32 오디오를 16-bit PCM으로 변환한다."""
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        return (audio * 32767).astype(np.int16)

    def _to_audio_result(self, audio: NDArray) -> AudioResult:
        """float32 오디오를 AudioResult로 변환한다."""
        return AudioResult(sr=self._hps.data.sampling_rate, data=self._to_pcm(audio))

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

    def optimize(self) -> None:
        """GPU 추론 속도를 최적화한다 (torch.compile).

        ``torch.compile(mode="reduce-overhead")`` 를 적용하여
        CUDA Graphs + Triton 커널 퓨전으로 10-25% 추론 속도를 향상시킨다.
        반복 추론하는 서버 환경에서 권장한다.

        일반적으로 :meth:`TTS.optimize` 를 통해 로드된 전체 화자를
        한 번에 최적화하는 것이 편리하다.

        .. note::

            첫 ``generate()`` 호출 시 컴파일 워밍업(1-2초)이 발생한다.
            1회성 추론에서는 워밍업 비용이 절감보다 클 수 있다.

        Raises:
            ValueError: CUDA가 아닌 디바이스에서 호출한 경우.
        """
        if "cuda" not in self._device:
            raise ValueError(
                "torch.compile은 CUDA 디바이스에서만 사용 가능합니다. "
                f"현재 디바이스: {self._device}"
            )

        import torch

        net_g = self._ensure_pytorch_model()
        torch.set_float32_matmul_precision("high")
        self._net_g = torch.compile(net_g, mode="reduce-overhead")
        self._backend = "compiled"
        logger.info(f"Speaker '{self.name}' → torch.compile backend")

    def __repr__(self) -> str:
        return f"Speaker('{self.name}', backend='{self._backend}', styles={list(self._style2id.keys())})"


def _split_sentences(text: str) -> list[str]:
    """텍스트를 문장 경계(。！？!?\\n)에서 분할한다."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]
