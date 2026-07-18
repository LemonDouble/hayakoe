"""HayaKoe 추론 엔진의 사용자 대면 클래스."""

from __future__ import annotations

import threading
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


# 소스 루트 아래 아티팩트 경로. pre_download 와 prepare/_init_bert 가 같은
# 위치를 봐야 오프라인 계약이 유지되므로 반드시 여기서만 정의한다.
_TOKENIZER_PREFIX = "bert/tokenizer"
_BERT_PREFIX = {"onnx": "onnx/bert/q8", "pytorch": "pytorch/bert/fp32"}


def _backend_for(device: str) -> str:
    """device 문자열로 백엔드를 판정한다 (``"pytorch"`` | ``"onnx"``)."""
    return "pytorch" if device.startswith("cuda") else "onnx"


def _speaker_prefix(backend: str, name: str) -> str:
    """소스 루트 아래 화자 아티팩트 경로."""
    return f"{backend}/speakers/{name}"


class TTS:
    """HayaKoe 추론 엔진.

    ``device`` 에 따라 백엔드를 자동 선택한다:
      - CPU → ONNX Runtime (Q8 BERT + FP32 Synthesizer)
      - CUDA → PyTorch FP32 + ``torch.compile``

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

        # load() 로 등록된 lazy 소스 — prepare() 전까지 아무것도 내려받지 않는다.
        self._specs: dict[str, Source] = {}
        self._speakers: dict[str, "Speaker"] = {}  # noqa: F821
        self._prepared: bool = False
        self._backend: Optional[str] = None
        self._bert_session = None
        # load()/prepare() 의 셋업 상태를 보호하는 락. 준비 단계 전용이며
        # generate() 등 서빙 경로는 이 락을 사용하지 않는다 (per-speaker 락 별개).
        self._lock = threading.Lock()

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
        with self._lock:
            self._specs[speaker_name] = parsed
        return self

    # ──────────────────────────── 실행 ────────────────────────────

    def prepare(self, *, warmup: bool = False, compile: bool = False) -> "TTS":
        """등록된 모든 화자 + BERT 를 다운로드하고 메모리에 올린다.

        - CPU: ONNX BERT Q8 세션 + 각 화자의 ONNX 세션을 만든다.
        - CUDA: PyTorch BERT FP32 + 화자 Synthesizer 로드 (eager).

        Args:
            warmup: CUDA 백엔드에서 더미 추론을 선행해 cuDNN 초기화 등 첫
                요청 지연을 prepare 단계로 옮긴다. ``compile=True`` 일 때는
                BERT 컴파일 비용도 여기서 지불된다. 서빙 시나리오에 권장.
                CPU 백엔드에서는 무시된다. 기본 ``False``.
            compile: CUDA 백엔드에서 공용 BERT 에 ``torch.compile`` 을
                적용한다. 실측 기준 워밍업 약 80초를 대가로 steady-state
                문장당 ~40ms (~20%) 빨라진다. 장기 실행 서버라면 켤 가치가
                있고, 스크립트/대화형 사용이라면 끄는 편이 낫다.
                CPU 백엔드에서는 무시된다. 기본 ``False``.

        캐시에 이미 있는 파일은 재사용한다. 아직 materialize 되지 않은 화자만
        새로 올리므로, ``load()`` 로 화자를 추가한 뒤 다시 호출하면 그 화자가
        추가로 준비된다 (이미 준비된 화자는 건너뛴다). ``load()`` 와 같은 락을
        공유하여 멀티스레드에서 동시에 호출해도 안전하다.
        Returns: ``self`` (체이닝용).
        """
        with self._lock:
            # 백엔드 결정 + BERT 초기화는 최초 1회만 (bert_models 는 멱등하지만
            # 불필요한 재확인을 피한다).
            if self._backend is None:
                backend = _backend_for(self._device)
                if backend == "pytorch":
                    self._validate_cuda()

                    import torch

                    # TF32 matmul 허용 (Ampere+). eager 경로도 혜택을 받으며
                    # 음질 영향은 무시 가능한 수준.
                    torch.set_float32_matmul_precision("high")
                self._backend = backend
                self._init_bert()

            # 아직 materialize 되지 않은 화자만 추린다. 락 안에서 스냅샷을
            # 만들므로 순회 중 load() 가 _specs 를 변경해도 안전하다.
            pending = [
                (name, source)
                for name, source in self._specs.items()
                if name not in self._speakers
            ]
            for name, source in pending:
                self._materialize_speaker(name, source)

            new_speakers = [self._speakers[name] for name, _ in pending]
            if self._backend == "pytorch":
                if compile:
                    self._compile()
                if warmup and new_speakers:
                    self._warmup(new_speakers)

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

        pyopenjtalk 사전도 함께 받아 둔다 (wheel 미포함 — 안 받아 두면
        런타임 첫 합성이 GitHub 에서 사전을 다운로드하게 된다).

        이후 런타임에서 동일한 ``cache_dir`` 로 ``TTS().load(...).prepare()``
        를 호출하면 캐시에서 즉시 로드된다. GPU 가 빌드 환경에 없어도 되며,
        BERT 가중치는 모델 자체에 올라가지 않는다.
        """
        backend = _backend_for(device)

        self._bert_source.fetch(_BERT_PREFIX[backend])
        self._bert_source.fetch(_TOKENIZER_PREFIX)

        for name, source in self._specs.items():
            source.fetch(_speaker_prefix(backend, name))

        # pyopenjtalk 사전(~22MB)은 첫 g2p 호출 시 site-packages 로 다운로드
        # 된다. 여기서 한 번 호출해 빌드 레이어에 미리 박아 둔다.
        import pyopenjtalk

        pyopenjtalk.g2p("あ")

        logger.info(
            f"Pre-downloaded ({backend}) → {self._cache_dir} "
            f"[{len(self._specs)} speakers]"
        )
        return self

    # ──────────────────────────── 기타 API ────────────────────────────

    def add_word(
        self, *, surface: str, reading: str, accent: int = 0, priority: int = 8
    ) -> None:
        """TTS 용 커스텀 단어 발음을 등록한다.

        Args:
            surface: 텍스트에 나타나는 단어 (예: ``"担々麺"``).
            reading: 가타카나 읽기 (예: ``"タンタンメン"``).
            accent: 피치가 내려가는 모라 위치 (0 = 평판/악센트 없음).
            priority: 등록 우선도 (0~10, 기본 8). 값이 클수록 기본 읽기보다
                우선한다. 강한 기본 읽기를 가진 단어는 낮으면 무시될 수 있다.
        """
        apply_word(
            surface=surface,
            pronunciation=reading,
            accent_type=accent,
            priority=priority,
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

        tok_dir = self._bert_source.fetch(_TOKENIZER_PREFIX)
        if not bert_models.is_tokenizer_loaded():
            bert_models.load_tokenizer(
                pretrained_model_name_or_path=str(tok_dir),
            )

        if self._backend == "onnx":
            import onnxruntime as ort

            onnx_dir = self._bert_source.fetch(_BERT_PREFIX["onnx"])
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
            model_dir = self._bert_source.fetch(_BERT_PREFIX["pytorch"])
            if bert_models.is_model_loaded():
                bert_models.transfer_model(self._device)
            else:
                bert_models.load_model(
                    pretrained_model_name_or_path=str(model_dir),
                    device=self._device,
                )

    def _materialize_speaker(self, name: str, source: Source) -> None:
        from hayakoe.api.speaker import Speaker

        model_dir = source.fetch(_speaker_prefix(self._backend, name))

        speaker = Speaker(
            name=name,
            model_dir=Path(model_dir),
            device=self._device,
            backend=self._backend,
            bert_session=self._bert_session,
        )
        self._speakers[name] = speaker

    def _compile(self) -> None:
        """공용 BERT 에 torch.compile 을 적용한다.

        Synthesizer(net_g) 는 컴파일하지 않는다: 추론은 ``forward`` 가 아닌
        ``infer`` 커스텀 메서드 경유라 ``torch.compile(net_g)`` 는 dynamo 를
        우회하는 no-op 이고, 메서드를 직접 컴파일해도 실측상 워밍업 +3.5분
        (문장 길이별 재컴파일) 대비 이득이 ~20% 에 그쳐 수지가 맞지 않는다.
        BERT 는 ``__call__`` 경유라 컴파일이 실제로 적용되며 문장당 ~40ms
        (~20%) 를 번다. ``bert_models.compile_model()`` 은 멱등하므로 반복
        호출해도 중첩 래핑이 쌓이지 않는다.
        """
        from hayakoe.nlp import bert_models

        bert_models.compile_model()

    def _warmup(self, speakers: list["Speaker"]) -> None:  # noqa: F821
        """주어진 화자에 대해 더미 추론을 돌려 첫 요청 지연을 prepare
        시점으로 옮긴다 (cuDNN/lazy 초기화, ``compile=True`` 라면 BERT
        컴파일 비용 포함).

        단일 문장 + 다중 문장 2종으로 돌려서 ``net_g.infer`` 와
        ``net_g.predict_durations`` 경로를 모두 데운다. 길이가 다른 입력
        2개는 BERT 컴파일 시 Dynamo ``automatic_dynamic`` 을 트리거해
        이후 다른 길이 요청의 full re-trace 도 피한다.

        실패해도 prepare 는 계속 진행.
        """
        import time

        samples = [
            "こんにちは、テストです。",  # 단일 문장 → net_g.infer
            "こんにちは。テストを始めます。",  # 다중 문장 → predict_durations + batch bert + infer
        ]
        for speaker in speakers:
            style = next(iter(speaker._style2id.keys()))
            t0 = time.perf_counter()
            try:
                for sample in samples:
                    speaker.generate(sample, style=style)
            except Exception as e:
                logger.warning(
                    f"Warmup failed for speaker '{speaker.name}': {e}"
                )
                continue
            logger.info(
                f"Speaker '{speaker.name}' warmed up in "
                f"{time.perf_counter() - t0:.1f}s"
            )
