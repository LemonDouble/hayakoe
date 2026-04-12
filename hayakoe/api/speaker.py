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
_MIN_PAUSE_SEC = 0.08  # 문장 간 최소 무음 보장 (80ms)
_SILENCE_WINDOW_MS = 10  # 무음 측정 윈도우 (ms)


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
        """텍스트에서 음성을 생성한다.

        여러 문장이 포함된 텍스트는 문장 경계(。！？!?\\n)에서
        자동 분할하여 개별 추론 후 연결한다. PyTorch/compiled 백엔드에서는
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
        bp = self._predict_pauses(
            text, sentences, style, style_weight,
            speaker_id, speed, sdp_ratio, noise_w,
        )

        if not batch_bert:
            parts: list[NDArray] = []
            for i, sentence in enumerate(sentences):
                audio = self._synthesize_one(sentence, **kwargs)
                if i > 0:
                    trailing = _measure_trailing_silence(parts[-1], sr)
                    gap = _make_pause_gap(
                        trailing, sr, np.float32, _pause_target(bp, i - 1),
                    )
                    if len(gap) > 0:
                        parts.append(gap)
                parts.append(audio)
            return self._to_audio_result(np.concatenate(parts))

        # 다중 문장: BERT 배치 추론 + 순차 합성
        style_vec = self._get_style_vector(style, style_weight)

        nlp_results = [self._preprocess_nlp(s) for s in sentences]

        if self._backend == "onnx":
            bert_features = self._batch_bert_onnx(nlp_results)
        else:
            bert_features = self._batch_bert_pytorch(nlp_results)

        parts: list[NDArray] = []
        for i, (nlp, ja_bert) in enumerate(zip(nlp_results, bert_features)):
            if i > 0:
                trailing = _measure_trailing_silence(parts[-1], sr)
                gap = _make_pause_gap(
                    trailing, sr, np.float32, _pause_target(bp, i - 1),
                )
                if len(gap) > 0:
                    parts.append(gap)
            _, phone_seq, tone_seq, lang_seq, _ = nlp

            if self._backend == "onnx":
                audio = self._synth_with_features_onnx(
                    phone_seq, tone_seq, lang_seq, ja_bert, style_vec,
                    speaker_id, speed, sdp_ratio, noise, noise_w,
                )
            else:
                audio = self._synth_with_features_pytorch(
                    phone_seq, tone_seq, lang_seq, ja_bert, style_vec,
                    speaker_id, speed, sdp_ratio, noise, noise_w,
                )

            if pitch_scale != 1.0 or intonation_scale != 1.0:
                _, audio = adjust_voice(
                    fs=sr, wave=audio,
                    pitch_scale=pitch_scale,
                    intonation_scale=intonation_scale,
                )
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

        PyTorch/compiled 백엔드에서는 시작 전 Duration Predictor로 전체
        텍스트의 문장 경계 무음 길이를 예측하여 각 문장 사이 gap에 반영한다.
        ONNX 백엔드는 최소 80ms 폴백을 사용한다.

        Args:
            text: 합성할 텍스트.
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
        prev_trailing = 0.0

        bp = self._predict_pauses(
            text, sentences, style, style_weight,
            speaker_id, speed, sdp_ratio, noise_w,
        )

        for i, sentence in enumerate(sentences):
            audio = self._synthesize_one(
                sentence, lang=lang, style=style, speaker_id=speaker_id,
                speed=speed, sdp_ratio=sdp_ratio, noise=noise, noise_w=noise_w,
                pitch_scale=pitch_scale, intonation_scale=intonation_scale,
                style_weight=style_weight,
            )
            trailing = _measure_trailing_silence(audio, sr)
            pcm = self._to_pcm(audio)

            if i > 0:
                gap = _make_pause_gap(
                    prev_trailing, sr, np.int16, _pause_target(bp, i - 1),
                )
                if len(gap) > 0:
                    pcm = np.concatenate([gap, pcm])

            prev_trailing = trailing
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

    # ── BERT 배치 추론 ──

    def _preprocess_nlp(self, text: str) -> tuple:
        """NLP 전처리 (BERT 제외): (norm_text, phone_seq, tone_seq, lang_seq, word2ph)."""
        from hayakoe.models import commons
        from hayakoe.nlp import clean_text_with_given_phone_tone, cleaned_text_to_sequence

        hps = self._hps
        norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
            text, Languages.JP,
            use_jp_extra=hps.version.endswith("JP-Extra"),
            raise_yomi_error=False,
        )
        phone_seq, tone_seq, lang_seq = cleaned_text_to_sequence(phone, tone, Languages.JP)

        if hps.data.add_blank:
            phone_seq = commons.intersperse(phone_seq, 0)
            tone_seq = commons.intersperse(tone_seq, 0)
            lang_seq = commons.intersperse(lang_seq, 0)
            for i in range(len(word2ph)):
                word2ph[i] *= 2
            word2ph[0] += 1

        return norm_text, phone_seq, tone_seq, lang_seq, word2ph

    def _batch_bert_pytorch(self, nlp_results: list[tuple]) -> list:
        """PyTorch BERT 배치 추론."""
        import torch

        from hayakoe.nlp import bert_models
        from hayakoe.nlp.japanese.g2p import text_to_sep_kata

        device = self._device
        model = bert_models.load_model(device=device)
        bert_models.transfer_model(device)
        tokenizer = bert_models.load_tokenizer()

        clean_texts = [
            "".join(text_to_sep_kata(nlp[0], raise_yomi_error=False)[0])
            for nlp in nlp_results
        ]

        with torch.no_grad():
            inputs = tokenizer(clean_texts, return_tensors="pt", padding=True)
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            res = model(**inputs, output_hidden_states=True)
            hidden = torch.cat(res["hidden_states"][-3:-2], -1).float()

        bert_features = []
        for i, (_, _, _, _, word2ph) in enumerate(nlp_results):
            clean_text = clean_texts[i]
            assert len(word2ph) == len(clean_text) + 2, clean_text
            feat = []
            for j in range(len(word2ph)):
                feat.append(hidden[i][j].repeat(word2ph[j], 1))
            bert_features.append(torch.cat(feat, dim=0).T)

        return bert_features

    def _batch_bert_onnx(self, nlp_results: list[tuple]) -> list:
        """ONNX BERT 배치 추론."""
        from hayakoe.nlp import bert_models
        from hayakoe.nlp.japanese.g2p import text_to_sep_kata

        tokenizer = bert_models.load_tokenizer()

        clean_texts = [
            "".join(text_to_sep_kata(nlp[0], raise_yomi_error=False)[0])
            for nlp in nlp_results
        ]

        inputs = tokenizer(clean_texts, return_tensors="np", padding=True)
        res = self._bert_session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })[0]

        bert_features = []
        for i, (_, _, _, _, word2ph) in enumerate(nlp_results):
            clean_text = clean_texts[i]
            assert len(word2ph) == len(clean_text) + 2, clean_text
            feat = []
            for j in range(len(word2ph)):
                feat.append(np.tile(res[i][j], (word2ph[j], 1)))
            bert_features.append(np.concatenate(feat, axis=0).T)

        return bert_features

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
            l = torch.LongTensor(lang_seq).to(device).unsqueeze(0)
            b = ja_bert.to(device).unsqueeze(0)
            sv = torch.from_numpy(style_vec).to(device).unsqueeze(0)
            sid_t = torch.LongTensor([sid]).to(device)

            output = net_g.infer(
                x, x_len, sid_t, t, l, b,
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
        l = np.array(lang_seq, dtype=np.int64)[np.newaxis, :]
        b = ja_bert[np.newaxis, :, :].astype(np.float32)
        s = style_vec[np.newaxis, :].astype(np.float32)
        sid_arr = np.array([sid], dtype=np.int64)

        output = self._synth_session.run(None, {
            "x": x, "x_lengths": x_len, "sid": sid_arr,
            "tone": t, "language": l, "bert": b, "style_vec": s,
            "noise_scale": np.array([noise], dtype=np.float32),
            "length_scale": np.array([speed], dtype=np.float32),
            "noise_scale_w": np.array([noise_w], dtype=np.float32),
            "sdp_ratio": np.array([sdp_ratio], dtype=np.float32),
        })
        return output[0][0, 0]

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

    def _predict_pauses(
        self, text: str, sentences: list[str], style: str, style_weight: float,
        speaker_id: int, speed: float, sdp_ratio: float, noise_w: float,
    ) -> Optional[list[float]]:
        """Duration predictor로 문장 경계 pause를 예측한다.

        PyTorch/compiled 백엔드는 항상 동작하고, ONNX 백엔드는
        ``duration_predictor.onnx`` 가 모델 디렉터리에 있으면 동작한다.
        예측 불가 시 ``None`` 을 반환한다 (호출 측에서 80ms 폴백 사용).
        """
        if len(sentences) <= 1:
            return None

        sv = self._get_style_vector(style, style_weight)

        if self._backend in ("pytorch", "compiled"):
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
        from hayakoe.models.infer_onnx import get_text_onnx
        from hayakoe.models.infer import (
            durations_to_boundary_pauses,
            find_boundary_punct_positions,
        )

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
        l = lang_ids[np.newaxis, :]
        b = bert[np.newaxis, :, :].astype(np.float32)
        s = style_vec[np.newaxis, :].astype(np.float32)
        sid_arr = np.array([sid], dtype=np.int64)

        durations = self._dp_session.run(None, {
            "x": x, "x_lengths": x_len, "sid": sid_arr,
            "tone": t, "language": l, "bert": b, "style_vec": s,
            "length_scale": np.array([length_scale], dtype=np.float32),
            "noise_scale_w": np.array([noise_scale_w], dtype=np.float32),
            "sdp_ratio": np.array([sdp_ratio], dtype=np.float32),
        })[0][0]  # [phone_len]

        return durations_to_boundary_pauses(
            durations, phone_list, punct_positions, num_sentences, self._hps,
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

        from hayakoe.nlp import bert_models

        net_g = self._ensure_pytorch_model()
        torch.set_float32_matmul_precision("high")
        self._net_g = torch.compile(net_g, mode="reduce-overhead")
        bert_models.compile_model()
        self._backend = "compiled"
        logger.info(f"Speaker '{self.name}' → torch.compile backend")

    def __repr__(self) -> str:
        return f"Speaker('{self.name}', backend='{self._backend}', styles={list(self._style2id.keys())})"


def _split_sentences(text: str) -> list[str]:
    """텍스트를 문장 경계(。！？!?\\n)에서 분할한다."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


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
