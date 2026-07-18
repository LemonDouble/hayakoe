from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import AsyncGenerator, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from hayakoe.api.audio_result import (
    AudioResult,
    StyleAccessor,
    streaming_wav_header,
)
from hayakoe.constants import Languages
from hayakoe.logging import logger
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.voice import adjust_voice


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])")
# 실제 발화되는(모라를 만드는) 문자: 히라가나/가타카나/한자/영숫자
_HAS_SPEECH_RE = re.compile(r"[ぁ-ゖァ-ヶ一-鿿々〆〇a-zA-Z0-9０-９Ａ-Ｚａ-ｚｦ-ﾟ]")


@dataclass(frozen=True)
class _SynthesisParams:
    """generate/stream 이 공유하는 합성 파라미터 묶음 (내부 전달용)."""

    lang: Union[str, Languages] = Languages.JP
    style: str = "Neutral"
    speaker_id: int = 0
    speed: float = 1.0
    sdp_ratio: float = 0.2
    noise: float = 0.6
    noise_w: float = 0.8
    pitch_scale: float = 1.0
    intonation_scale: float = 1.0
    style_weight: float = 1.0


def _validate_lang(lang: Union[str, Languages]) -> None:
    """현재 일본어만 지원한다 — JP 이외 값은 진입점에서 명시적으로 거부."""
    value = Languages(lang.value if hasattr(lang, "value") else str(lang))
    if value != Languages.JP:
        raise NotImplementedError(
            f"Only Japanese (Languages.JP) is supported, got: {value}"
        )


_MIN_PAUSE_SEC = 0.08  # 문장 간 최소 무음 보장 (80ms)
_SILENCE_WINDOW_MS = 10  # 무음 측정 윈도우 (ms)


if TYPE_CHECKING:
    from hayakoe.models.models_jp_extra import (
        SynthesizerTrn as SynthesizerTrnJPExtra,
    )


class Speaker:
    """materialize된 화자 모델.

    직접 생성하지 말고 :meth:`TTS.load` + :meth:`TTS.prepare` 로 얻어야 한다.
    준비가 끝나면 ``tts.speakers["name"]`` 에서 가져온다.

    백엔드별 동작:

    - **onnx** (CPU): ONNX Runtime 으로 추론. ``TTS(device="cpu")`` 에서 자동 선택.
    - **pytorch** (CUDA): PyTorch eager. ``TTS(device="cuda")`` 에서 자동 선택.
      ``prepare(compile=True)`` 로 공용 BERT 에 ``torch.compile`` 을 적용해
      문장당 ~20% 를 추가로 얻을 수 있다 (워밍업 ~80초 필요).

    **Thread safety** — 각 Speaker 인스턴스는 내부 ``threading.Lock`` 으로
    ``generate`` / ``stream`` 호출을 직렬화한다. FastAPI 같은 싱글톤 서빙
    환경에서 여러 요청이 동시에 들어와도 안전하게 동작한다 (다른 Speaker
    끼리는 병렬 가능). BERT 는 공용 리소스이나 thread-safe 한 경로만
    사용하므로 별도 lock 이 없다.

    사용법::

        from hayakoe import TTS

        tts = TTS(device="cpu").load("jvnv-F1-jp").prepare()
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
        # generate/stream 호출 직렬화용 per-Speaker lock
        self._gen_lock = threading.Lock()

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

        # pytorch 백엔드는 지연 로드, onnx 는 아래에서 즉시 초기화
        self._net_g: Optional[SynthesizerTrnJPExtra] = None
        self._synth_session = None
        self._dp_session = None

        if backend == "onnx":
            self._init_onnx_synth()
            self._init_onnx_duration_predictor()

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

    def _init_onnx_duration_predictor(self):
        """ONNX Duration Predictor 세션을 생성한다 (선택적).

        문장 경계 무음 길이 예측에만 사용되며, 파일이 없으면 폴백
        (고정 80ms pause) 으로 동작한다.
        """
        import onnxruntime as ort

        onnx_path = self._model_dir / "duration_predictor.onnx"
        if not onnx_path.exists():
            return  # 옵션이므로 조용히 폴백

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._dp_session = ort.InferenceSession(
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

    @property
    def sampling_rate(self) -> int:
        """모델 샘플링 레이트 (Hz)."""
        return self._hps.data.sampling_rate

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
        batch_bert: bool = True,
    ) -> AudioResult:
        """텍스트에서 음성을 생성한다 (thread-safe).

        여러 문장이 포함된 텍스트는 문장 경계(。！？!?\\n)에서
        자동 분할하여 개별 추론 후 연결한다. PyTorch 백엔드에서는
        전체 텍스트를 Duration Predictor로 한 번 돌려 문장 경계의 자연스러운
        무음 길이를 예측하고, 이를 문장 사이 gap에 반영한다. ONNX 백엔드는
        최소 80ms 폴백을 사용한다.

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
            batch_bert: 다중 문장 텍스트에서 문장별 BERT 추론을 한 번의
                배치로 처리한다 (기본 ``True``). ``False`` 면 문장마다
                개별 추론한다 (배치 padding 과 수치가 미세하게 다를 수
                있어 검증/비교용).

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
        _validate_lang(lang)
        params = _SynthesisParams(
            lang=lang, style=style, speaker_id=speaker_id,
            speed=speed, sdp_ratio=sdp_ratio, noise=noise, noise_w=noise_w,
            pitch_scale=pitch_scale, intonation_scale=intonation_scale,
            style_weight=style_weight,
        )
        with self._gen_lock:
            return self._generate_locked(text, params, batch_bert=batch_bert)

    async def agenerate(
        self,
        text: str,
        **kwargs,
    ) -> AudioResult:
        """비동기 래퍼 — 별도 스레드에서 :meth:`generate` 를 실행한다.

        FastAPI 같은 async 핸들러에서 호출하면 이벤트 루프를 블록하지 않는다.
        """
        return await asyncio.to_thread(self.generate, text, **kwargs)

    def _generate_locked(
        self,
        text: str,
        params: _SynthesisParams,
        *,
        batch_bert: bool = True,
    ) -> AudioResult:
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return self._to_audio_result(self._synthesize_one(text, params))

        sr = self._hps.data.sampling_rate
        bp = self._predict_pauses(
            text, sentences, params.style, params.style_weight,
            params.speaker_id, params.speed, params.sdp_ratio, params.noise_w,
        )

        # 다중 문장: BERT 배치 추론 (기본) 후 순차 합성. batch_bert=False 는
        # BERT 준비만 문장별 개별 추론으로 바뀌고 합성 루프는 공유한다.
        if batch_bert:
            style_vec = self._get_style_vector(params.style, params.style_weight)
            nlp_results = [self._preprocess_nlp(s) for s in sentences]
            if self._backend == "onnx":
                bert_features = self._batch_bert_onnx(nlp_results)
            else:
                bert_features = self._batch_bert_pytorch(nlp_results)

        parts: list[NDArray] = []
        for i, sentence in enumerate(sentences):
            if i > 0:
                trailing = _measure_trailing_silence(parts[-1], sr)
                gap = _boundary_gap(trailing, sr, np.float32, bp, i - 1)
                if len(gap) > 0:
                    parts.append(gap)
            if batch_bert:
                audio = self._synthesize_one_with_features(
                    nlp_results[i], bert_features[i], style_vec, params,
                )
            else:
                audio = self._synthesize_one(sentence, params)
            parts.append(audio)

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
    ) -> Generator[AudioResult, None, None]:
        """텍스트를 문장 단위로 스트리밍 생성한다.

        문장 경계(。！？!?\\n)에서 분할하여 각 문장을 개별 추론하고,
        완료된 순서대로 yield한다. 첫 문장이 완성되는 즉시
        재생을 시작할 수 있어 체감 지연이 줄어든다.

        PyTorch 백엔드에서는 시작 전 Duration Predictor로 전체
        텍스트의 문장 경계 무음 길이를 예측하여 각 문장 사이 gap에 반영한다.
        ONNX 백엔드는 최소 80ms 폴백을 사용한다.

        Args:
            text: 합성할 텍스트.
            lang 이하: :meth:`generate` 와 동일한 파라미터
                (``batch_bert`` 는 받지 않는다 — 전체 문장 BERT 배치는
                첫 청크 지연을 키워 스트리밍 목적에 반한다).

        Yields:
            문장별 :class:`AudioResult`. 두 번째 문장부터 앞에 무음이 포함된다.

        Example::

            for chunk in speaker.stream("こんにちは。元気ですか？"):
                play(chunk.to_bytes())  # 문장별로 바로 재생

        **Thread safety** — 제너레이터가 살아있는 동안 per-speaker lock 을
        보유한다. ``close()`` 또는 소진되면 해제되므로 ``for`` 문으로 돌리거나
        ``try/finally`` 안에서 사용해야 다른 요청이 차단되지 않는다.
        """
        _validate_lang(lang)
        params = _SynthesisParams(
            lang=lang, style=style, speaker_id=speaker_id,
            speed=speed, sdp_ratio=sdp_ratio, noise=noise, noise_w=noise_w,
            pitch_scale=pitch_scale, intonation_scale=intonation_scale,
            style_weight=style_weight,
        )
        self._gen_lock.acquire()
        try:
            yield from self._stream_locked(text, params)
        finally:
            self._gen_lock.release()

    async def astream(
        self,
        text: str,
        **kwargs,
    ) -> AsyncGenerator[AudioResult, None]:
        """비동기 스트리밍 — 전용 단일 워커 스레드에서 :meth:`stream` 의 각 chunk 를 받아온다.

        FastAPI ``StreamingResponse`` 와 조합해 바로 스트리밍할 수 있다.
        제너레이터 소진 또는 async iterator close 시 lock 이 해제된다.

        스트림 하나가 워커 스레드 하나를 수명 내내 전용한다. asyncio 공유
        기본 executor 를 청크마다 빌리는 방식은 대기 스트림들이 풀을 다
        점유하는 순간 lock 보유자가 다음 청크를 꺼낼 스레드를 얻지 못해
        앱 전체 ``to_thread`` 가 영구 정지하는 데드락이 있었다.
        """
        gen = self.stream(text, **kwargs)
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        _SENTINEL = object()
        try:
            while True:
                chunk = await loop.run_in_executor(executor, next, gen, _SENTINEL)
                if chunk is _SENTINEL:
                    return
                yield chunk
        finally:
            # close 를 같은 워커 스레드에 제출한다. 단일 워커가 next() 와
            # close() 를 자연 직렬화하므로, 합성 중 취소되어도 실행 중인
            # 제너레이터를 닫으려다 나는 ValueError: generator already
            # executing 이 발생할 수 없고, 진행 중이던 문장이 끝나는 즉시
            # lock 이 결정적으로 해제된다 (기존에는 GC 시점까지 잔류).
            executor.submit(gen.close)
            executor.shutdown(wait=False)

    def stream_wav(
        self,
        text: str,
        **kwargs,
    ) -> Generator[bytes, None, None]:
        """하나의 유효한 WAV 스트림을 bytes 청크로 yield 한다.

        첫 청크는 길이 미정 스트리밍 WAV 헤더
        (:func:`~hayakoe.api.audio_result.streaming_wav_header`), 이후 청크는
        문장별 raw PCM 이다. 전부 이어붙이면 표준 플레이어가 끝까지 재생하는
        올바른 WAV 스트림이 된다 (:meth:`stream` + ``to_bytes()`` 이어붙이기는
        첫 문장만 재생되므로 스트리밍 응답에는 이 메서드를 쓸 것).

        Args:
            text: 합성할 텍스트.
            **kwargs: :meth:`stream` 과 동일한 파라미터.

        **Thread safety** — :meth:`stream` 과 동일하게 제너레이터가 살아있는
        동안 per-speaker lock 을 보유한다.
        """
        yield streaming_wav_header(self.sampling_rate)
        for chunk in self.stream(text, **kwargs):
            yield chunk.pcm_bytes()

    async def astream_wav(
        self,
        text: str,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """:meth:`stream_wav` 의 비동기 버전.

        FastAPI ``StreamingResponse`` 에 그대로 넘길 수 있다::

            return StreamingResponse(
                speaker.astream_wav(text), media_type="audio/wav"
            )
        """
        yield streaming_wav_header(self.sampling_rate)
        async for chunk in self.astream(text, **kwargs):
            yield chunk.pcm_bytes()

    def _stream_locked(
        self,
        text: str,
        params: _SynthesisParams,
    ) -> Generator[AudioResult, None, None]:
        sentences = _split_sentences(text)
        if not sentences:
            return

        sr = self._hps.data.sampling_rate
        prev_trailing = 0.0
        # running-max 게인: 문장마다 자기 피크로 정규화하면 조용한 문장이
        # 만피크로 부풀려져 문장 간 상대 음량이 무너지고 generate() 와도
        # 달라진다. 지금까지 본 최대 피크로 나눠 (게인은 내려가기만 함)
        # 이미 내보낸 청크를 소급 조정하지 않으면서 다이내믹스를 보존한다.
        max_peak = 0.0

        bp = self._predict_pauses(
            text, sentences, params.style, params.style_weight,
            params.speaker_id, params.speed, params.sdp_ratio, params.noise_w,
        )

        for i, sentence in enumerate(sentences):
            audio = self._synthesize_one(sentence, params)
            trailing = _measure_trailing_silence(audio, sr)
            max_peak = max(max_peak, float(np.abs(audio).max()))
            pcm = self._to_pcm(audio, peak=max_peak)

            if i > 0:
                gap = _boundary_gap(prev_trailing, sr, np.int16, bp, i - 1)
                if len(gap) > 0:
                    pcm = np.concatenate([gap, pcm])

            prev_trailing = trailing
            yield AudioResult(sr=sr, data=pcm)

    def _synthesize_one(self, text: str, params: _SynthesisParams) -> NDArray:
        """단일 텍스트 → float32 오디오 배열."""
        style_vec = self._get_style_vector(params.style, params.style_weight)

        if self._backend == "onnx":
            audio = self._generate_onnx(
                text, style_vec, params.speaker_id,
                params.speed, params.sdp_ratio, params.noise, params.noise_w,
            )
        else:
            audio = self._generate_pytorch(
                text, style_vec, params.speaker_id,
                params.speed, params.sdp_ratio, params.noise, params.noise_w,
            )

        if params.pitch_scale != 1.0 or params.intonation_scale != 1.0:
            _, audio = adjust_voice(
                fs=self._hps.data.sampling_rate,
                wave=audio,
                pitch_scale=params.pitch_scale,
                intonation_scale=params.intonation_scale,
            )

        return audio

    @staticmethod
    def _to_pcm(audio: NDArray, peak: Optional[float] = None) -> NDArray[np.int16]:
        """float32 오디오를 16-bit PCM으로 변환한다.

        peak 를 주면 자기 피크 대신 그 값으로 정규화한다. 스트리밍에서
        running-max 게인으로 문장 간 상대 음량을 보존하는 데 쓴다.
        """
        if peak is None:
            peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak
        return (audio * 32767).astype(np.int16)

    def _to_audio_result(self, audio: NDArray) -> AudioResult:
        """float32 오디오를 AudioResult로 변환한다."""
        return AudioResult(sr=self._hps.data.sampling_rate, data=self._to_pcm(audio))

    # ── BERT 배치 추론 ──

    def _preprocess_nlp(self, text: str) -> tuple:
        """NLP 전처리 (BERT 제외): (norm_text, phone_seq, tone_seq, lang_seq, word2ph)."""
        from hayakoe.models.text_preprocess import prepare_phone_sequences

        return prepare_phone_sequences(text, self._hps)

    @staticmethod
    def _bert_clean_texts(nlp_results: list[tuple]) -> list[str]:
        """BERT 입력용 클린 텍스트 (norm_text → sep_kata 결합)."""
        from hayakoe.nlp.japanese.g2p import text_to_sep_kata

        return [
            "".join(text_to_sep_kata(nlp[0], raise_yomi_error=False)[0])
            for nlp in nlp_results
        ]

    @staticmethod
    def _expand_bert_features(
        hidden: NDArray, nlp_results: list[tuple], clean_texts: list[str],
    ) -> list[NDArray]:
        """토큰 단위 hidden 을 word2ph 에 따라 음소 단위로 확장한다."""
        features = []
        for i, (_, _, _, _, word2ph) in enumerate(nlp_results):
            clean_text = clean_texts[i]
            assert len(word2ph) == len(clean_text) + 2, clean_text
            features.append(np.repeat(hidden[i][: len(word2ph)], word2ph, axis=0).T)
        return features

    def _batch_bert_pytorch(self, nlp_results: list[tuple]) -> list:
        """PyTorch BERT 배치 추론."""
        import torch

        from hayakoe.nlp import bert_models

        device = self._device
        model = bert_models.load_model(device=device)
        bert_models.transfer_model(device)
        tokenizer = bert_models.load_tokenizer()

        clean_texts = self._bert_clean_texts(nlp_results)

        with torch.no_grad():
            inputs = tokenizer(clean_texts, return_tensors="pt", padding=True)
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            res = model(**inputs, output_hidden_states=True)
            hidden = res["hidden_states"][-3].float()

        return [
            torch.from_numpy(feat)
            for feat in self._expand_bert_features(
                hidden.cpu().numpy(), nlp_results, clean_texts,
            )
        ]

    def _batch_bert_onnx(self, nlp_results: list[tuple]) -> list:
        """ONNX BERT 배치 추론."""
        from hayakoe.nlp import bert_models

        tokenizer = bert_models.load_tokenizer()

        clean_texts = self._bert_clean_texts(nlp_results)

        inputs = tokenizer(clean_texts, return_tensors="np", padding=True)
        res = self._bert_session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })[0]

        return self._expand_bert_features(res, nlp_results, clean_texts)

    def _synth_with_features_pytorch(
        self, phone_seq, tone_seq, lang_seq, ja_bert, style_vec,
        sid, speed, sdp_ratio, noise, noise_w,
    ) -> NDArray:
        """pre-computed BERT 특징으로 PyTorch 합성."""
        import torch

        net_g = self._ensure_pytorch_model()
        device = self._device

        with torch.no_grad():
            phones = torch.LongTensor(phone_seq)
            x = phones.to(device).unsqueeze(0)
            x_len = torch.LongTensor([phones.size(0)]).to(device)
            t = torch.LongTensor(tone_seq).to(device).unsqueeze(0)
            lang = torch.LongTensor(lang_seq).to(device).unsqueeze(0)
            b = ja_bert.to(device).unsqueeze(0)
            sv = torch.from_numpy(style_vec).to(device).unsqueeze(0)
            sid_t = torch.LongTensor([sid]).to(device)

            output = net_g.infer(
                x, x_len, sid_t, t, lang, b,
                style_vec=sv, length_scale=speed,
                sdp_ratio=sdp_ratio, noise_scale=noise, noise_scale_w=noise_w,
            )
            return output[0][0, 0].data.cpu().float().numpy()

    def _synth_with_features_onnx(
        self, phone_seq, tone_seq, lang_seq, ja_bert, style_vec,
        sid, speed, sdp_ratio, noise, noise_w,
    ) -> NDArray:
        """pre-computed BERT 특징으로 ONNX 합성."""
        x = np.array(phone_seq, dtype=np.int64)[np.newaxis, :]
        x_len = np.array([len(phone_seq)], dtype=np.int64)
        t = np.array(tone_seq, dtype=np.int64)[np.newaxis, :]
        lang = np.array(lang_seq, dtype=np.int64)[np.newaxis, :]
        b = ja_bert[np.newaxis, :, :].astype(np.float32)
        s = style_vec[np.newaxis, :].astype(np.float32)
        sid_arr = np.array([sid], dtype=np.int64)

        output = self._synth_session.run(None, {
            "x": x, "x_lengths": x_len, "sid": sid_arr,
            "tone": t, "language": lang, "bert": b, "style_vec": s,
            "noise_scale": np.array([noise], dtype=np.float32),
            "length_scale": np.array([speed], dtype=np.float32),
            "noise_scale_w": np.array([noise_w], dtype=np.float32),
            "sdp_ratio": np.array([sdp_ratio], dtype=np.float32),
        })
        return output[0][0, 0]

    def _synthesize_one_with_features(
        self, nlp: tuple, ja_bert, style_vec: NDArray, params: _SynthesisParams,
    ) -> NDArray:
        """pre-computed BERT 특징으로 단일 문장 → float32 오디오 배열."""
        _, phone_seq, tone_seq, lang_seq, _ = nlp

        if self._backend == "onnx":
            audio = self._synth_with_features_onnx(
                phone_seq, tone_seq, lang_seq, ja_bert, style_vec,
                params.speaker_id, params.speed, params.sdp_ratio,
                params.noise, params.noise_w,
            )
        else:
            audio = self._synth_with_features_pytorch(
                phone_seq, tone_seq, lang_seq, ja_bert, style_vec,
                params.speaker_id, params.speed, params.sdp_ratio,
                params.noise, params.noise_w,
            )

        if params.pitch_scale != 1.0 or params.intonation_scale != 1.0:
            _, audio = adjust_voice(
                fs=self._hps.data.sampling_rate,
                wave=audio,
                pitch_scale=params.pitch_scale,
                intonation_scale=params.intonation_scale,
            )
        return audio

    # ── 단일 문장 추론 ──

    def _generate_onnx(self, text, style_vec, sid, speed, sdp_ratio, noise, noise_w):
        from hayakoe.models.infer_onnx import infer_onnx

        return infer_onnx(
            text=text,
            style_vec=style_vec,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noise_w,
            length_scale=speed,
            sid=sid,
            hps=self._hps,
            bert_session=self._bert_session,
            synth_session=self._synth_session,
        )

    def _generate_pytorch(self, text, style_vec, sid, speed, sdp_ratio, noise, noise_w):
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
                hps=self._hps,
                net_g=net_g,
                device=self._device,
                style_vec=style_vec,
            )

    # ── 문장 경계 무음 예측 ──

    def _predict_pauses(
        self, text: str, sentences: list[str], style: str, style_weight: float,
        speaker_id: int, speed: float, sdp_ratio: float, noise_w: float,
    ) -> Optional[list[float]]:
        """Duration predictor로 문장 경계 pause를 예측한다.

        PyTorch 백엔드는 항상 동작하고, ONNX 백엔드는
        ``duration_predictor.onnx`` 가 모델 디렉터리에 있으면 동작한다.
        예측 불가 시 ``None`` 을 반환한다 (호출 측에서 80ms 폴백 사용).
        """
        if len(sentences) <= 1:
            return None

        sv = self._get_style_vector(style, style_weight)

        if self._backend == "pytorch":
            from hayakoe.models.infer import predict_boundary_pauses

            return predict_boundary_pauses(
                text=text, style_vec=sv, length_scale=speed,
                sid=speaker_id, num_sentences=len(sentences),
                hps=self._hps, net_g=self._ensure_pytorch_model(),
                device=self._device,
                sdp_ratio=sdp_ratio, noise_scale_w=noise_w,
            )

        if self._backend == "onnx" and self._dp_session is not None:
            return self._predict_pauses_onnx(
                text, sv, speed, speaker_id, len(sentences), sdp_ratio, noise_w,
            )

        return None

    def _predict_pauses_onnx(
        self, text: str, style_vec: NDArray, length_scale: float,
        sid: int, num_sentences: int, sdp_ratio: float, noise_scale_w: float,
    ) -> Optional[list[float]]:
        """ONNX duration predictor로 문장 경계 pause를 예측한다."""
        # torch-free 모듈에서 가져온다. infer.py 는 top-level 에서 torch 를
        # import 하므로, torch 없는 CPU 환경에서 이 경로가 크래시하지 않도록
        # 순수 헬퍼는 boundary_pauses 에서 직접 import 한다.
        from hayakoe.models.boundary_pauses import (
            durations_to_boundary_pauses,
            find_boundary_punct_positions,
        )
        from hayakoe.models.infer_onnx import get_text_onnx

        bert, phones, tones, lang_ids = get_text_onnx(
            text, self._hps, self._bert_session,
        )
        phone_list = phones.tolist()
        punct_positions = find_boundary_punct_positions(phone_list)
        if not punct_positions:
            return []

        x = phones[np.newaxis, :]
        x_len = np.array([len(phones)], dtype=np.int64)
        t = tones[np.newaxis, :]
        lang = lang_ids[np.newaxis, :]
        b = bert[np.newaxis, :, :].astype(np.float32)
        s = style_vec[np.newaxis, :].astype(np.float32)
        sid_arr = np.array([sid], dtype=np.int64)

        durations = self._dp_session.run(None, {
            "x": x, "x_lengths": x_len, "sid": sid_arr,
            "tone": t, "language": lang, "bert": b, "style_vec": s,
            "length_scale": np.array([length_scale], dtype=np.float32),
            "noise_scale_w": np.array([noise_scale_w], dtype=np.float32),
            "sdp_ratio": np.array([sdp_ratio], dtype=np.float32),
        })[0][0]  # [phone_len]

        return durations_to_boundary_pauses(
            durations, phone_list, punct_positions, num_sentences, self._hps,
        )

    def __repr__(self) -> str:
        return f"Speaker('{self.name}', backend='{self._backend}', styles={list(self._style2id.keys())})"


def _split_sentences(text: str) -> list[str]:
    """텍스트를 문장 경계(。！？!?\\n)에서 분할한다.

    발화 문자가 없는 조각(구두점만 있는 것)은 독립된 문장이 아니다. 그대로 두면
    합성 시 무음/무성음이 되어 낭비되거나 f0 처리에서 오류를 낸다. 그래서 그런
    조각은 직전 문장에 합치고(맨 앞이면 다음 문장 앞에 붙인다), 결과적으로 모든
    문장이 발화 내용을 갖도록 한다.
    """
    fragments = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    sentences: list[str] = []
    pending = ""  # 아직 문장이 없을 때 앞에 모아둔 구두점
    for frag in fragments:
        if not _HAS_SPEECH_RE.search(frag):
            if sentences:
                sentences[-1] += frag
            else:
                pending += frag
        else:
            sentences.append(pending + frag)
            pending = ""
    if pending:  # 전부 구두점뿐인 텍스트
        if sentences:
            sentences[-1] += pending
        else:
            sentences.append(pending)
    return sentences


def _measure_trailing_silence(audio: NDArray, sr: int) -> float:
    """합성된 오디오 끝부분의 무음 길이를 초 단위로 측정한다."""
    window = max(1, int(sr * _SILENCE_WINDOW_MS / 1000))
    if len(audio) < window:
        return 0.0

    peak = float(np.abs(audio).max())
    if peak == 0:
        return len(audio) / sr

    threshold = peak * 0.02  # 피크의 2% 이하를 무음으로 간주

    pos = len(audio)
    while pos >= window:
        chunk = audio[pos - window : pos]
        if float(np.abs(chunk).max()) > threshold:
            break
        pos -= window
    return (len(audio) - pos) / sr


def _pause_target(
    boundary_pauses: Optional[list[float]], boundary_idx: int,
) -> float:
    """예측된 문장 경계 pause 목록에서 해당 경계의 목표 무음 길이를 조회한다.

    예측값이 없거나 (ONNX 백엔드) 범위 밖이면 최소 pause를 반환한다.
    """
    if boundary_pauses and 0 <= boundary_idx < len(boundary_pauses):
        return boundary_pauses[boundary_idx]
    return _MIN_PAUSE_SEC


def _make_pause_gap(
    trailing_sec: float, sr: int, dtype: type, target_sec: float = _MIN_PAUSE_SEC,
) -> NDArray:
    """트레일링 무음을 고려해 추가 무음 샘플을 생성한다.

    ``target_sec`` 이 주어지면 해당 목표까지 부족분만 보충한다.
    모델이 이미 충분한 무음을 생성했으면 빈 배열을 반환한다.
    """
    target = max(target_sec, _MIN_PAUSE_SEC)
    if trailing_sec >= target:
        return np.array([], dtype=dtype)
    extra = target - trailing_sec
    return np.zeros(int(sr * extra), dtype=dtype)


def _boundary_gap(
    trailing_sec: float,
    sr: int,
    dtype: type,
    boundary_pauses: Optional[list[float]],
    boundary_idx: int,
) -> NDArray:
    """문장 경계에 삽입할 무음 gap 을 만든다 (예측 pause 조회 + 부족분 보충)."""
    return _make_pause_gap(
        trailing_sec, sr, dtype, _pause_target(boundary_pauses, boundary_idx),
    )
