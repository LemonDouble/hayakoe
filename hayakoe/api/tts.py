from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from hayakoe.constants import HF_REPO
from hayakoe.logging import logger
from hayakoe.nlp.japanese.user_dict import apply_word, update_dict
from hayakoe.utils.strenum import StrEnum


class Lang(StrEnum):
    """공개 API용 언어 enum."""

    JA = "JP"  # 내부 코드에서는 "JP" 사용


class TTS:
    """HayaKoe 추론 엔진.

    device에 따라 최적 백엔드를 자동 선택한다:
      - CPU → ONNX Runtime (Q8 BERT + FP32 Synth)
      - CUDA → PyTorch FP32

    사용법::

        from hayakoe import TTS
        speaker = TTS().load("jvnv-F1-jp")
        speaker.generate("こんにちは").save("output.wav")
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ) -> None:
        """TTS 엔진을 초기화한다. 필요 시 모델을 다운로드하고 로드한다.

        Args:
            device: "cpu" (ONNX 자동) 또는 "cuda" (PyTorch 자동).
            cache_dir: HuggingFace 캐시 디렉토리.
        """
        self._device = device
        self._cache_dir = cache_dir
        self._speakers: dict[str, "Speaker"] = {}

        if "cuda" in device:
            try:
                import torch  # noqa: F401
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "GPU 추론에는 PyTorch(CUDA)가 필요합니다. 설치:\n"
                    "  pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
                    "  pip install hayakoe[gpu]"
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA를 사용할 수 없습니다. CUDA 드라이버와 PyTorch CUDA 빌드를 확인하세요.\n"
                    f"  torch version: {torch.__version__}\n"
                    f"  torch.cuda.is_available(): {torch.cuda.is_available()}"
                )
            self._backend = "pytorch"
            self._bert_session = None
            self._init_pytorch(device, cache_dir)
        else:
            self._backend = "onnx"
            self._bert_session = self._init_onnx()

        logger.info(f"TTS ready — {self._backend} on {device}")

    def _init_pytorch(self, device: str, cache_dir: Optional[str]) -> None:
        """PyTorch 백엔드 초기화: BERT 모델 + 토크나이저 로드."""
        from hayakoe.api.resources import load_bert

        load_bert(device=device, cache_dir=cache_dir)

    def _init_onnx(self):
        """ONNX 백엔드 초기화: BERT Q8 세션 생성 + 토크나이저 로드."""
        import onnxruntime as ort

        from hayakoe.nlp import bert_models

        # 토크나이저 로드 (ONNX에서도 텍스트 전처리에 필요)
        if not bert_models.is_tokenizer_loaded():
            bert_models.load_tokenizer()

        # BERT ONNX Q8 다운로드 + 세션 생성
        bert_path = _download_bert_onnx()
        logger.info(f"Loading ONNX BERT from {bert_path.name}...")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(bert_path), sess_opts, providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def prepare(cache_dir: Optional[str] = None) -> None:
        """모델을 사전 다운로드한다. GPU 불필요.

        Docker 빌드나 CI에서 사용::

            RUN python -c "from hayakoe import TTS; TTS.prepare()"
        """
        _download_bert_onnx()
        logger.info("Models downloaded.")

    def load(
        self,
        name: str,
        model_dir: Union[str, Path, None] = None,
    ) -> "Speaker":
        """화자 모델을 로드한다.

        이름만 지정하면 HuggingFace에서 자동 다운로드한다.

        Args:
            name: 화자 이름 (예: "jvnv-F1-jp").
            model_dir: 로컬 모델 디렉토리. 생략 시 HF에서 다운로드.

        Returns:
            .generate() 메서드를 가진 Speaker 인스턴스.
        """
        from hayakoe.api.speaker import Speaker

        if model_dir is None:
            model_dir = _download_speaker(name, self._backend)

        speaker = Speaker(
            name=name,
            model_dir=Path(model_dir),
            device=self._device,
            backend=self._backend,
            bert_session=self._bert_session,
        )
        self._speakers[name] = speaker
        return speaker

    def add_word(self, *, surface: str, reading: str, accent: int = 0) -> None:
        """TTS용 커스텀 단어 발음을 등록한다.

        Args:
            surface: 텍스트에 나타나는 단어 (예: "担々麺").
            reading: 가타카나 읽기 (예: "タンタンメン").
            accent: 피치가 내려가는 모라 위치 (0 = 평판/악센트 없음).
        """
        apply_word(
            surface=surface,
            pronunciation=reading,
            accent_type=accent,
        )

    @property
    def speakers(self) -> dict[str, "Speaker"]:
        """현재 로드된 화자 목록."""
        return dict(self._speakers)

    def __repr__(self) -> str:
        names = list(self._speakers.keys())
        return f"TTS(device='{self._device}', backend='{self._backend}', speakers={names})"


# ──────────────────────── 다운로드 헬퍼 ────────────────────────


def _download_bert_onnx() -> Path:
    """ONNX BERT Q8 모델을 다운로드하고 경로를 반환한다."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(HF_REPO, allow_patterns=["onnx/bert/q8/*"])
    return Path(local) / "onnx" / "bert" / "q8" / "bert_q8.onnx"


def _download_speaker(name: str, backend: str) -> Path:
    """HuggingFace에서 화자 모델을 다운로드하고 로컬 경로를 반환한다."""
    from huggingface_hub import snapshot_download

    if backend == "onnx":
        prefix = "onnx/speakers"
    else:
        prefix = "pytorch/speakers"

    logger.info(f"Downloading speaker '{name}' ({backend})...")
    local = snapshot_download(HF_REPO, allow_patterns=[f"{prefix}/{name}/*"])
    speaker_dir = Path(local) / prefix / name

    if not speaker_dir.exists():
        raise FileNotFoundError(
            f"Speaker '{name}' not found in {HF_REPO}. "
            f"Available: jvnv-F1-jp, jvnv-F2-jp, jvnv-M1-jp, jvnv-M2-jp"
        )
    return speaker_dir
