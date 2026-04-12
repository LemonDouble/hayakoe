"""HayaKoe 추론 엔진의 사용자 대면 클래스."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from hayakoe.api.sources import Source, parse_source
from hayakoe.constants import (
    DEFAULT_BERT_SOURCE,
    DEFAULT_SPEAKER_SOURCE,
    default_cache_dir,
)
from hayakoe.logging import logger
from hayakoe.nlp.japanese.user_dict import apply_word
from hayakoe.utils.strenum import StrEnum


class Lang(StrEnum):
    """공개 API용 언어 enum."""

    JA = "JP"  # 내부 코드에서는 "JP" 사용


@dataclass
class _SpeakerSpec:
    """load() 시 등록되는 lazy 스펙 — prepare() 전까지 아무것도 내려받지 않는다."""

    name: str
    source: Source


class TTS:
    """HayaKoe 추론 엔진.

    ``device`` 에 따라 백엔드를 자동 선택한다:
      - CPU → ONNX Runtime (Q8 BERT + FP32 Synthesizer)
      - CUDA → PyTorch FP32 + ``torch.compile(reduce-overhead)``

    기본 흐름 — **체이닝 지원**::

        from hayakoe import TTS

        tts = (
            TTS(device="cuda")
            .load("jvnv-F1-jp")                       # 공식 HF repo
            .load("my-speaker", source="hf://me/priv") # 내 비공개 repo
            .prepare()                                  # 실제 다운로드 + 로드 + compile
        )

        speaker = tts.speakers["jvnv-F1-jp"]
        speaker.generate("こんにちは").save("out.wav")

    Docker 빌드에서는 GPU 없이 미리 받기만 할 수 있다::

        # Dockerfile 빌드 단계 — CUDA 불필요
        TTS().load("jvnv-F1-jp").pre_download(device="cuda")

        # 런타임 이미지 — 캐시에서 즉시 로드
        TTS(device="cuda").load("jvnv-F1-jp").prepare()

    캐시 위치는 ``cache_dir`` 또는 ``HAYAKOE_CACHE`` env 로 제어하며,
    기본값은 ``$CWD/hayakoe_cache`` 다. HuggingFace / S3 / 로컬 모든 소스가
    같은 루트 아래에 저장된다.
    """

    def __init__(
        self,
        device: str = "cpu",
        *,
        cache_dir: Union[str, Path, None] = None,
        bert_source: str = DEFAULT_BERT_SOURCE,
        hf_token: Optional[str] = None,
    ) -> None:
        """TTS 엔진 스펙을 등록한다 (실제 로드는 :meth:`prepare` 에서).

        Args:
            device: ``"cpu"`` → ONNX, ``"cuda"`` → PyTorch. ``__init__`` 시점에는
                검증하지 않고 ``prepare()`` 또는 ``pre_download()`` 에서 확인한다.
            cache_dir: 로컬 캐시 루트. 기본 ``$CWD/hayakoe_cache``.
            bert_source: BERT 공용 리소스 소스 URI.
                기본 ``hf://lemondouble/hayakoe`` — 자체 재배포가 필요할 때
                ``s3://...`` 또는 ``hf://your/repo`` 로 덮어쓸 수 있다.
            hf_token: private HuggingFace repo 접근용 토큰.
        """
        self._device = device
        self._cache_dir = (
            Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir()
        )
        self._hf_token = hf_token
        self._bert_source = parse_source(
            bert_source, self._cache_dir, token=hf_token,
        )

        self._specs: dict[str, _SpeakerSpec] = {}
        self._speakers: dict[str, "Speaker"] = {}  # noqa: F821
        self._prepared: bool = False
        self._backend: Optional[str] = None
        self._bert_session = None

    # ──────────────────────────── 등록 ────────────────────────────

    def load(
        self,
        speaker_name: str,
        *,
        source: Optional[str] = None,
    ) -> "TTS":
        """화자를 등록한다. 체이닝 가능 — 실제 fetch 는 :meth:`prepare` 에서.

        Args:
            speaker_name: prefix 아래 있는 화자 이름 (예: ``"jvnv-F1-jp"``).
            source: 화자를 담은 소스 URI. 미지정 시 공식 repo
                (``hf://lemondouble/hayakoe``). 자체 repo 를 쓰려면
                ``hf://your/repo`` / ``s3://bucket/prefix`` / ``file:///path``
                / 로컬 절대경로 중 하나를 넘긴다. 실제 파일은 소스 루트 아래
                ``{onnx|pytorch}/speakers/{speaker_name}/`` 에서 찾는다.
        """
        uri = source or DEFAULT_SPEAKER_SOURCE
        parsed = parse_source(uri, self._cache_dir, token=self._hf_token)
        self._specs[speaker_name] = _SpeakerSpec(name=speaker_name, source=parsed)
        return self

    # ──────────────────────────── 실행 ────────────────────────────

    def prepare(self) -> "TTS":
        """등록된 모든 화자 + BERT 를 다운로드하고 메모리에 올린다.

        - CPU: ONNX BERT Q8 세션 + 각 화자의 ONNX 세션을 만든다.
        - CUDA: PyTorch BERT FP32 + 화자 Synthesizer 로드 + ``torch.compile``
          자동 적용.

        캐시에 이미 있는 파일은 재사용한다. 반복 호출은 no-op.
        Returns: ``self`` (체이닝용).
        """
        if self._prepared:
            return self

        if "cuda" in self._device:
            self._validate_cuda()
            self._backend = "pytorch"
        else:
            self._backend = "onnx"

        self._init_bert()

        for name, spec in self._specs.items():
            self._materialize_speaker(name, spec)

        if self._device.startswith("cuda") and self._speakers:
            self._compile_all()

        self._prepared = True
        names = list(self._speakers.keys())
        logger.info(
            f"TTS ready — {self._backend} on {self._device}, speakers={names}"
        )
        return self

    def pre_download(self, device: str = "cuda") -> "TTS":
        """등록된 자원을 로컬 캐시에 다운로드만 한다 (메모리 로드 X).

        Docker 빌드 단계용. ``device`` 는 어떤 백엔드용 아티팩트를 받을지만 결정:

        - ``"cpu"`` → ONNX BERT Q8 + ``onnx/speakers/<name>/*``
        - ``"cuda"`` → PyTorch BERT FP32 + ``pytorch/speakers/<name>/*``

        이후 런타임에서 동일한 ``cache_dir`` 로 ``TTS().load(...).prepare()``
        를 호출하면 캐시에서 즉시 로드된다. GPU 가 빌드 환경에 없어도 되며,
        BERT 가중치는 모델 자체에 올라가지 않는다.
        """
        backend = "pytorch" if device.startswith("cuda") else "onnx"

        if backend == "onnx":
            self._bert_source.fetch("onnx/bert/q8")
            self._bert_source.fetch("bert/tokenizer")
        else:
            self._bert_source.fetch("pytorch/bert/fp32")
            self._bert_source.fetch("bert/tokenizer")

        for name, spec in self._specs.items():
            spec.source.fetch(f"{backend}/speakers/{name}")

        logger.info(
            f"Pre-downloaded ({backend}) → {self._cache_dir} "
            f"[{len(self._specs)} speakers]"
        )
        return self

    # ──────────────────────────── 기타 API ────────────────────────────

    def add_word(self, *, surface: str, reading: str, accent: int = 0) -> None:
        """TTS 용 커스텀 단어 발음을 등록한다.

        Args:
            surface: 텍스트에 나타나는 단어 (예: ``"担々麺"``).
            reading: 가타카나 읽기 (예: ``"タンタンメン"``).
            accent: 피치가 내려가는 모라 위치 (0 = 평판/악센트 없음).
        """
        apply_word(
            surface=surface,
            pronunciation=reading,
            accent_type=accent,
        )

    @property
    def speakers(self) -> dict[str, "Speaker"]:  # noqa: F821
        """준비된 화자 dict. ``prepare()`` 전 접근 시 예외."""
        if not self._prepared:
            raise RuntimeError(
                "tts.prepare() 를 먼저 호출하세요 — 화자가 아직 materialize 되지 않았습니다."
            )
        return dict(self._speakers)

    def __repr__(self) -> str:
        state = "prepared" if self._prepared else "pending"
        names = list(self._specs.keys())
        return (
            f"TTS(device='{self._device}', state={state}, "
            f"speakers={names})"
        )

    # ──────────────────────────── 내부 헬퍼 ────────────────────────────

    @staticmethod
    def _validate_cuda() -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "GPU 추론에는 PyTorch(CUDA)가 필요합니다. 설치:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
                "  pip install hayakoe[gpu]"
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다. 드라이버와 PyTorch CUDA 빌드를 확인하세요.\n"
                f"  torch version: {torch.__version__}\n"
                f"  torch.cuda.is_available(): {torch.cuda.is_available()}"
            )

    def _init_bert(self) -> None:
        from hayakoe.nlp import bert_models

        tok_dir = self._bert_source.fetch("bert/tokenizer")
        if not bert_models.is_tokenizer_loaded():
            bert_models.load_tokenizer(
                pretrained_model_name_or_path=str(tok_dir),
            )

        if self._backend == "onnx":
            import onnxruntime as ort

            onnx_dir = self._bert_source.fetch("onnx/bert/q8")
            onnx_path = onnx_dir / "bert_q8.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"bert_q8.onnx not found under {onnx_dir}"
                )
            logger.info(f"Loading ONNX BERT from {onnx_path.name}...")
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._bert_session = ort.InferenceSession(
                str(onnx_path), sess_opts, providers=["CPUExecutionProvider"],
            )
        else:
            model_dir = self._bert_source.fetch("pytorch/bert/fp32")
            if bert_models.is_model_loaded():
                bert_models.transfer_model(self._device)
            else:
                bert_models.load_model(
                    pretrained_model_name_or_path=str(model_dir),
                    device=self._device,
                )

    def _materialize_speaker(self, name: str, spec: _SpeakerSpec) -> None:
        from hayakoe.api.speaker import Speaker

        backend_prefix = "pytorch" if "cuda" in self._device else "onnx"
        model_dir = spec.source.fetch(f"{backend_prefix}/speakers/{name}")

        speaker = Speaker(
            name=name,
            model_dir=Path(model_dir),
            device=self._device,
            backend=self._backend,
            bert_session=self._bert_session,
        )
        self._speakers[name] = speaker

    def _compile_all(self) -> None:
        """모든 Speaker + 공용 BERT 에 torch.compile 을 적용한다."""
        from hayakoe.nlp import bert_models

        for speaker in self._speakers.values():
            speaker._apply_compile()
        bert_models.compile_model()
