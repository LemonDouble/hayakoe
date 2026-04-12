"""화자 배포 (Publish) 인터랙티브 메뉴.

학습이 끝난 화자를 HuggingFace / S3 / 로컬 경로에 올려, 런타임에서::

    TTS(device=...).load("my-voice", source="hf://me/private-voices").prepare()

처럼 다운로드 받아 쓸 수 있게 한다. 레포 루트 아래 구조는 다음을 따른다::

    pytorch/speakers/<name>/{config.json, style_vectors.npy, *.safetensors}
    onnx/speakers/<name>/{config.json, style_vectors.npy, synthesizer.onnx,
                          duration_predictor.onnx}

CPU(ONNX) 백엔드를 고를 때 ONNX 아티팩트가 없으면 내부적으로 ONNX 내보내기를
즉시 수행한다. 유저가 원한다면 미리 준비한 커스텀 safetensors 파일도 쓸 수
있다. 배포가 끝나면 실제로 hayakoe 런타임으로 같은 주소에서 다운로드해 추론
테스트까지 돌려본다.
"""

import difflib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.syntax import Syntax

from cli.publish.credentials import (
    _write_env_var,
    ensure_hf_token,
    ensure_s3_credentials,
    load_env_file,
)
from cli.training.dataset import DatasetInfo, discover_datasets
from cli.ui.console import console
from cli.ui.prompts import confirm, edit_value, select_from_list, text_input

from hayakoe.api.sources import (
    HFSource,
    LocalSource,
    S3Source,
    normalize_hf_uri,
    parse_source,
)


# ──────────────────────────── PublishSource ────────────────────────────


@dataclass
class PublishSource:
    """업로드 대상 화자의 물리적 소스. 학습된 dataset 이든, 유저가 직접 준비한
    폴더 (config.json + style_vectors.npy + *.safetensors 를 포함하는
    "hayakoe 포맷" 디렉토리) 든 동일하게 취급하기 위한 추상화.

    Attributes:
        display_name: 화자 선택 메뉴에 표시될 이름.
        default_target_name: ``edit_value("배포 이름", ...)`` 의 기본값.
        config_path: ``config.json`` 절대 경로. ``num_styles`` / ``style2id``
            가 반드시 포함돼 있어야 런타임에서 style 로딩이 정확해짐.
        style_vectors_path: ``style_vectors.npy`` 절대 경로.
        checkpoints: 선택 가능한 ``.safetensors`` 목록 (mtime 오름차순).
        existing_onnx_dir: 이미 ``synthesizer.onnx`` 가 있는 디렉토리
            (있으면 재사용). ``None`` 이면 export 필요.
        onnx_export_dir: export 가 필요한 경우 내보낼 디렉토리. dataset
            소스는 ``<ds.path>/onnx`` 로 캐시하고, 폴더 소스는 tempdir 을
            써서 사용자 폴더를 오염시키지 않는다.
    """

    display_name: str
    default_target_name: str
    config_path: Path
    style_vectors_path: Path
    checkpoints: list[Path]
    existing_onnx_dir: Optional[Path]
    onnx_export_dir: Path


def _publish_source_from_dataset(ds: DatasetInfo) -> Optional[PublishSource]:
    """학습된 dataset → PublishSource. 학습 산출물이 부족하면 ``None``."""
    sbv2_config = ds.data_dir / "config.json"
    if not sbv2_config.exists():
        return None
    try:
        model_name = json.loads(sbv2_config.read_text()).get("model_name", "")
    except Exception:
        return None
    if not model_name:
        return None

    exports = ds.path / "exports" / model_name
    if not exports.exists():
        return None

    # exports/config.json 에 num_styles / style2id 가 들어있어 런타임 style
    # 로딩이 정확함 — sbv2_data/config.json 은 학습 시 스냅샷이라 그게 없다.
    exports_config = exports / "config.json"
    config_path = exports_config if exports_config.exists() else sbv2_config

    style_vectors = exports / "style_vectors.npy"
    if not style_vectors.exists():
        return None

    checkpoints = sorted(exports.glob("*.safetensors"))
    if not checkpoints:
        return None

    onnx_dir = ds.path / "onnx"
    existing_onnx = onnx_dir if (onnx_dir / "synthesizer.onnx").exists() else None

    return PublishSource(
        display_name=ds.name,
        default_target_name=ds.name,
        config_path=config_path,
        style_vectors_path=style_vectors,
        checkpoints=checkpoints,
        existing_onnx_dir=existing_onnx,
        onnx_export_dir=onnx_dir,
    )


def _publish_source_from_folder(folder: Path) -> PublishSource:
    """유저가 직접 선택한 폴더 → PublishSource. 포맷 검증 실패 시 ValueError.

    필수 파일: ``config.json``, ``style_vectors.npy``, 하나 이상의 ``*.safetensors``.
    선택: ``synthesizer.onnx`` / ``duration_predictor.onnx`` 가 있으면 재사용.
    """
    if not folder.is_dir():
        raise ValueError(f"폴더가 아닙니다: {folder}")

    config_path = folder / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json 이 없습니다: {folder}")

    style_vectors = folder / "style_vectors.npy"
    if not style_vectors.exists():
        raise ValueError(f"style_vectors.npy 가 없습니다: {folder}")

    checkpoints = sorted(folder.glob("*.safetensors"))
    if not checkpoints:
        raise ValueError(f".safetensors 파일이 없습니다: {folder}")

    existing_onnx = folder if (folder / "synthesizer.onnx").exists() else None

    return PublishSource(
        display_name=folder.name,
        default_target_name=folder.name,
        config_path=config_path,
        style_vectors_path=style_vectors,
        checkpoints=checkpoints,
        existing_onnx_dir=existing_onnx,
        # 사용자 폴더를 오염시키지 않도록 tempdir 에 내보냄
        onnx_export_dir=Path(tempfile.mkdtemp(prefix="hayakoe-onnx-")),
    )


# ──────────────────────────── 스테이징 ────────────────────────────


def _stage_pytorch(src: PublishSource, ckpt: Path, dst: Path) -> None:
    """선택된 safetensors + config + style_vectors.npy 를 dst 에 복사."""
    dst.mkdir(parents=True, exist_ok=True)
    for source_file, name in [
        (ckpt, ckpt.name),
        (src.style_vectors_path, "style_vectors.npy"),
        (src.config_path, "config.json"),
    ]:
        if not source_file.exists():
            raise FileNotFoundError(f"필수 파일 없음: {source_file}")
        shutil.copy2(source_file, dst / name)


def _stage_onnx(src: PublishSource, onnx_dir: Path, dst: Path) -> None:
    """synthesizer.onnx + (선택) duration_predictor.onnx + config + style vec."""
    dst.mkdir(parents=True, exist_ok=True)

    mandatory = [
        (onnx_dir / "synthesizer.onnx", "synthesizer.onnx"),
        (src.style_vectors_path, "style_vectors.npy"),
        (src.config_path, "config.json"),
    ]
    optional = [
        (onnx_dir / "duration_predictor.onnx", "duration_predictor.onnx"),
        (onnx_dir / "synthesizer.onnx.data", "synthesizer.onnx.data"),
    ]

    for source_file, name in mandatory:
        if not source_file.exists():
            raise FileNotFoundError(f"필수 파일 없음: {source_file}")
        shutil.copy2(source_file, dst / name)
    for source_file, name in optional:
        if source_file.exists():
            shutil.copy2(source_file, dst / name)


# ──────────────────────────── 목적지 프롬프트 ────────────────────────────


def _persist_if_changed(key: str, value: str) -> None:
    """값이 이전과 다르면 ``.env`` 에 조용히 저장."""
    if value and os.environ.get(key) != value:
        _write_env_var(key, value)
        console.print(f"  [dim]→ .env 에 {key} 저장[/dim]")


def _prompt_hf_destination() -> tuple[Optional[str], Optional[str]]:
    """HF repo 주소와 토큰을 받아 ``(hf:// URI, token)`` 을 반환한다."""
    load_env_file()
    default_repo = os.environ.get("HAYAKOE_HF_REPO", "")

    console.print()
    console.print(
        "  [dim]HF repo 주소 — 아래 형식 모두 허용:\n"
        "    lemondouble/hayakoe-voices\n"
        "    hf://lemondouble/hayakoe-voices\n"
        "    hf://lemondouble/hayakoe-voices@main\n"
        "    https://huggingface.co/lemondouble/hayakoe-voices\n"
        "    https://huggingface.co/lemondouble/hayakoe-voices/tree/dev[/dim]"
    )
    raw = text_input("HF repo 주소", default=default_repo).strip()
    if not raw:
        return None, None

    normalized = normalize_hf_uri(raw)
    if normalized is None:
        console.print(f"  [error]해석할 수 없는 HF 주소입니다: {raw}[/error]")
        return None, None

    token = ensure_hf_token()
    if not token:
        return None, None

    _persist_if_changed("HAYAKOE_HF_REPO", raw)
    return normalized, token


def _prompt_s3_destination() -> Optional[str]:
    """S3 bucket + prefix 를 받아 ``s3://...`` URI 를 반환한다."""
    load_env_file()
    if not ensure_s3_credentials():
        return None

    default_bucket = os.environ.get("HAYAKOE_S3_BUCKET", "")
    default_prefix = os.environ.get("HAYAKOE_S3_PREFIX", "")

    bucket = text_input("S3 Bucket", default=default_bucket).strip()
    if not bucket:
        return None
    prefix = text_input(
        "Prefix (루트면 Enter)", default=default_prefix,
    ).strip().strip("/")

    _persist_if_changed("HAYAKOE_S3_BUCKET", bucket)
    _persist_if_changed("HAYAKOE_S3_PREFIX", prefix)

    uri = f"s3://{bucket}"
    if prefix:
        uri = f"{uri}/{prefix}"
    return uri


def _prompt_local_destination() -> Optional[str]:
    """로컬 디렉토리 경로를 받아 ``file://`` URI 를 반환한다."""
    load_env_file()
    default_local = os.environ.get("HAYAKOE_LOCAL_PATH", "")

    raw = text_input(
        "로컬 디렉토리 절대 경로", default=default_local,
    ).strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    _persist_if_changed("HAYAKOE_LOCAL_PATH", str(path))
    return f"file://{path}"


# ──────────────────────────── 백엔드 선택 ────────────────────────────


def _select_backends() -> list[str]:
    """항상 CPU/GPU/CPU+GPU/뒤로 를 제공. 없는 아티팩트는 즉시 export 로 만든다."""
    choice = select_from_list(
        "어떤 백엔드를 배포할까요?",
        [
            "CPU (ONNX) — GPU 없는 서버/로컬용, BERT Q8 + ONNX Runtime",
            "GPU (PyTorch) — torch.compile + CUDA Graph, 최소 지연",
            "CPU + GPU (권장) — 양쪽 환경 모두 배포",
            "뒤로",
        ],
    )
    if choice == "뒤로":
        return []
    if choice.startswith("CPU + GPU"):
        return ["pytorch", "onnx"]
    if choice.startswith("GPU"):
        return ["pytorch"]
    return ["onnx"]


# ──────────────────────────── 체크포인트 선택 ────────────────────────────


def _select_checkpoint(src: PublishSource) -> Optional[Path]:
    """``src.checkpoints`` 중 하나 선택. 1개뿐이면 자동 선택."""
    if not src.checkpoints:
        console.print("  [error]선택 가능한 .safetensors 가 없습니다.[/error]")
        return None
    if len(src.checkpoints) == 1:
        only = src.checkpoints[0]
        console.print(f"  [dim]체크포인트 자동 선택 → {only.name}[/dim]")
        return only

    options = [c.name for c in src.checkpoints] + ["뒤로"]
    pick = select_from_list(
        "체크포인트 선택 (품질 리포트에서 확인한 최적 모델)", options,
    )
    if pick == "뒤로":
        return None
    return next(c for c in src.checkpoints if c.name == pick)


# ──────────────────────────── ONNX export (내부) ────────────────────────────


def _ensure_onnx_exports(src: PublishSource, ckpt: Path) -> Path:
    """해당 체크포인트로 synthesizer.onnx + duration_predictor.onnx 를 보장.

    ``src.existing_onnx_dir`` 가 있으면 재사용, 없으면 ``src.onnx_export_dir``
    에 내보낸 뒤 그 경로를 반환한다.
    """
    from cli.export.exporter import export_duration_predictor, export_synthesizer

    if src.existing_onnx_dir is not None:
        console.print(f"  [dim]기존 ONNX 재사용 → {src.existing_onnx_dir}[/dim]")
        return src.existing_onnx_dir

    target = src.onnx_export_dir
    console.print(f"  [dim]ONNX 내보내기 시작 → {target}[/dim]")
    export_synthesizer(src.config_path, ckpt, target)
    export_duration_predictor(src.config_path, ckpt, target)
    return target


# ──────────────────────────── README 자동 생성 ────────────────────────────


def _list_remote_speakers(source, token: Optional[str]) -> list[str]:
    """목적지의 ``pytorch/speakers/`` 와 ``onnx/speakers/`` 아래 화자 이름 합집합.

    실패 시 빈 리스트. README 동적 렌더링에서 "이 저장소가 현재 가지고 있는
    화자" 를 표기하기 위해 사용한다.
    """
    names: set[str] = set()
    try:
        if isinstance(source, LocalSource):
            for backend in ("pytorch", "onnx"):
                base = source.root / backend / "speakers"
                if base.is_dir():
                    for p in base.iterdir():
                        if p.is_dir():
                            names.add(p.name)
        elif isinstance(source, HFSource):
            from huggingface_hub import HfApi

            api = HfApi(token=token or source.token)
            files = api.list_repo_files(source.repo, revision=source.revision)
            for f in files:
                for backend in ("pytorch", "onnx"):
                    prefix = f"{backend}/speakers/"
                    if f.startswith(prefix):
                        tail = f[len(prefix):]
                        name = tail.split("/", 1)[0]
                        if name:
                            names.add(name)
        elif isinstance(source, S3Source):
            client = source._client()
            for backend in ("pytorch", "onnx"):
                key_prefix = source._key_prefix(f"{backend}/speakers")
                paginator = client.get_paginator("list_objects_v2")
                for page in paginator.paginate(
                    Bucket=source.bucket,
                    Prefix=key_prefix,
                    Delimiter="/",
                ):
                    for common in page.get("CommonPrefixes", []) or []:
                        p = common.get("Prefix", "")
                        rel = p[len(key_prefix):].rstrip("/")
                        if rel:
                            names.add(rel)
    except Exception:
        return []
    return sorted(names)


def _render_runtime_usage(destination_uri: str, speakers: list[str]) -> str:
    """README Runtime usage 섹션을 화자 리스트 기반으로 생성한다.

    - 0개 (fallback): ``<speaker-name>`` 플레이스홀더
    - 1개: 단일 체이닝 예제
    - 2개↑: ``.load()`` 체이닝 예제
    - 공통: Docker 빌드 시점 ``pre_download`` 예제 추가
    """
    display = speakers if speakers else ["<speaker-name>"]
    first = display[0]

    header = "**Available speakers:** " + ", ".join(f"`{n}`" for n in display)

    runtime_load_lines = [
        f'    .load("{name}", source="{destination_uri}")' for name in display
    ]
    runtime_block = [
        "```python",
        "from hayakoe import TTS",
        "",
        "tts = (",
        '    TTS(device="cuda")',
        *runtime_load_lines,
        "    .prepare()",
        ")",
        f'audio = tts.speakers["{first}"].generate("こんにちは")',
        'audio.save("out.wav")',
        "```",
    ]

    docker_load_lines = [
        f'    .load("{name}", source="{destination_uri}")' for name in display
    ]
    docker_block = [
        "Docker 빌드 단계에서 캐시만 미리 받아두려면 (GPU 불필요):",
        "",
        "```python",
        "# Dockerfile 빌드 스텝 — CUDA 없이 실행 가능",
        "from hayakoe import TTS",
        "",
        "(",
        "    TTS()",
        *docker_load_lines,
        '    .pre_download(device="cuda")',
        ")",
        "```",
    ]

    return "\n".join([header, "", *runtime_block, "", *docker_block])


_README_TEMPLATE = """\
---
license: other
language:
  - ja
  - ko
  - en
  - zh
library_name: hayakoe
tags:
  - tts
  - text-to-speech
  - style-bert-vits2
  - hayakoe
pipeline_tag: text-to-speech
---

# HayaKoe Speaker Repository

> Auto-generated by `hayakoe-dev publish`. 자동 생성되었습니다.

This repository hosts one or more speakers for the
[HayaKoe](https://github.com/lemondouble/hayakoe) TTS runtime.

## Runtime usage

{runtime_usage}

---

## 📂 Directory Layout / 디렉토리 구조 / ディレクトリ構造 / 目录结构

```
<repo-root>/
├── pytorch/
│   └── speakers/
│       └── <speaker-name>/
│           ├── config.json           # HyperParameters
│           ├── style_vectors.npy     # Style embeddings
│           └── *.safetensors         # Synthesizer checkpoint
└── onnx/
    └── speakers/
        └── <speaker-name>/
            ├── config.json
            ├── style_vectors.npy
            ├── synthesizer.onnx      # FP32 Synthesizer
            └── duration_predictor.onnx
```

- `pytorch/` — GPU 추론용 (CUDA + `torch.compile`)
- `onnx/` — CPU 추론용 (ONNX Runtime, BERT Q8)
- Each speaker is self-contained under its own `<speaker-name>/` directory.

---

## 🇰🇷 한국어

이 저장소는 HayaKoe TTS 런타임이 내려받아 쓰는 **화자 아티팩트** 모음입니다.

- `pytorch/speakers/<name>/` — GPU 추론용 (safetensors 체크포인트)
- `onnx/speakers/<name>/` — CPU 추론용 (ONNX Runtime)
- 각 화자는 자체 디렉토리 안에 config / style_vectors / 모델 파일을 모두 포함
  하여 독립적으로 로드됩니다.

런타임에서는 `TTS(...).load(name, source="{uri}").prepare()` 한 번이면
자동으로 이 주소에서 필요한 파일을 캐싱해 사용합니다.

## 🇺🇸 English

This repository stores **speaker artifacts** distributed to the HayaKoe TTS
runtime.

- `pytorch/speakers/<name>/` — GPU backend (safetensors)
- `onnx/speakers/<name>/` — CPU backend (ONNX Runtime)
- Each speaker is fully self-contained inside its own directory, so the
  runtime can load any speaker independently.

At inference time, `TTS(...).load(name, source="{uri}").prepare()` will fetch
and cache the required files from this location on demand.

## 🇯🇵 日本語

このリポジトリは、HayaKoe TTS ランタイムが読み込んで使用する **話者アセット**
のコレクションです。

- `pytorch/speakers/<name>/` — GPU 推論用 (safetensors)
- `onnx/speakers/<name>/` — CPU 推論用 (ONNX Runtime)
- 各話者のディレクトリ内に config / style_vectors / モデルファイルをすべて
  含めており、単独でロード可能です。

実行時には `TTS(...).load(name, source="{uri}").prepare()` のみで、必要な
ファイルをこの場所からキャッシュしつつロードします。

## 🇨🇳 中文

本仓库存放 HayaKoe TTS 运行时所使用的 **说话人资源**。

- `pytorch/speakers/<name>/` — GPU 推理 (safetensors)
- `onnx/speakers/<name>/` — CPU 推理 (ONNX Runtime)
- 每个说话人都被独立封装在各自目录中，包含 config、style_vectors 及模型文件。

运行时只需 `TTS(...).load(name, source="{uri}").prepare()` 即可自动从该位置
拉取并缓存所需文件。
"""


def _fetch_remote_readme(source, token: Optional[str]) -> Optional[str]:
    """목적지 루트의 README.md 내용을 문자열로 가져온다. 없거나 실패하면 None."""
    try:
        if isinstance(source, LocalSource):
            p = source.root / "README.md"
            return p.read_text(encoding="utf-8") if p.exists() else None

        if isinstance(source, S3Source):
            import boto3

            client = boto3.client("s3")
            key = f"{source.prefix}/README.md" if source.prefix else "README.md"
            try:
                obj = client.get_object(Bucket=source.bucket, Key=key)
                return obj["Body"].read().decode("utf-8")
            except Exception:
                return None

        if isinstance(source, HFSource):
            from huggingface_hub import hf_hub_download

            try:
                local = hf_hub_download(
                    source.repo,
                    "README.md",
                    revision=source.revision,
                    token=token or source.token,
                )
                return Path(local).read_text(encoding="utf-8")
            except Exception:
                return None
    except Exception:
        return None
    return None


def _print_readme_diff(existing: str, new_content: str, max_lines: int = 60) -> None:
    diff = list(difflib.unified_diff(
        existing.splitlines(),
        new_content.splitlines(),
        fromfile="remote/README.md",
        tofile="generated/README.md",
        lineterm="",
        n=2,
    ))
    shown = diff[:max_lines]
    console.print(
        Syntax("\n".join(shown), "diff", theme="ansi_dark", line_numbers=False)
    )
    if len(diff) > max_lines:
        console.print(f"  [dim]... ({len(diff) - max_lines} more lines)[/dim]")


def _maybe_stage_readme(
    staging: Path,
    source,
    destination_uri: str,
    token: Optional[str],
    target_name: str,
) -> None:
    """루트 README.md 상태에 따라 자동 생성본을 스테이징한다.

    - 없음 → 생성 여부 확인 후 스테이징
    - 동일 → 건너뛰기
    - 변경점 있음 → **자동으로 덮어쓰지 않음**, diff 공지 후 승인받아야 갱신

    Runtime usage 섹션은 목적지에 이미 있는 화자 리스트 + 방금 올릴
    ``target_name`` 을 합쳐서 실제 사용 가능한 모델 기준으로 렌더링한다.
    """
    remote_speakers = _list_remote_speakers(source, token)
    speaker_set = set(remote_speakers)
    speaker_set.add(target_name)
    speakers = sorted(speaker_set)

    runtime_usage = _render_runtime_usage(destination_uri, speakers)
    new_content = _README_TEMPLATE.format(
        uri=destination_uri, runtime_usage=runtime_usage,
    )
    existing = _fetch_remote_readme(source, token)

    if existing is None:
        console.print()
        console.print(
            "  [dim]목적지 루트에 README.md 가 없습니다. ko/en/ja/zh 로 작성된\n"
            f"  데이터 저장 포맷 설명 README.md 를 자동 생성해 함께 올릴 수 있습니다.\n"
            f"  (Runtime usage 에는 화자 {len(speakers)}개가 반영됩니다: "
            f"{', '.join(speakers)})[/dim]"
        )
        if not confirm("README.md 를 자동 생성해 함께 배포할까요?", default=True):
            return
        (staging / "README.md").write_text(new_content, encoding="utf-8")
        console.print("  [dim]stage → README.md (신규, ko/en/ja/zh)[/dim]")
        return

    if existing.strip() == new_content.strip():
        console.print("  [dim]README.md 최신 상태 — 건너뜁니다.[/dim]")
        return

    console.print()
    console.print(
        "  [warning]목적지에 이미 README.md 가 존재합니다 — 자동으로 덮어쓰지\n"
        "  않습니다. 아래 diff 를 확인하고 승인해야만 갱신됩니다. 거부하면\n"
        "  기존 README.md 는 그대로 유지됩니다.[/warning]"
    )
    console.print(
        f"  [dim]Runtime usage 에는 화자 {len(speakers)}개가 반영됩니다: "
        f"{', '.join(speakers)}[/dim]"
    )
    _print_readme_diff(existing, new_content)
    if not confirm(
        "README.md 를 새 템플릿으로 갱신할까요? (거부 시 원격 README 유지)",
        default=False,
    ):
        console.print("  [dim]README.md 갱신을 건너뜁니다 — 원격 README 는 그대로 유지됩니다.[/dim]")
        return
    (staging / "README.md").write_text(new_content, encoding="utf-8")
    console.print("  [dim]stage → README.md (갱신)[/dim]")


# ──────────────────────────── 업로드 후 검증 ────────────────────────────


_VERIFY_TEXT = "テスト音声です。"


def _reset_gpu_state_if_needed(device: str) -> None:
    """GPU 검증 사이에 글로벌 PyTorch 상태를 리셋한다.

    같은 프로세스에서 여러 번 ``TTS(device="cuda").prepare()`` 를 돌릴 때
    - 글로벌 BERT (``bert_models._loaded_model``) 에 중첩 ``torch.compile``
      래핑이 쌓이는 것
    - ``mode="reduce-overhead"`` CUDA Graphs 메모리 풀이 이전 compile 에
      남아 두 번째 캡처에서 충돌하는 것
    을 막기 위해 글로벌 모델을 해제하고 dynamo / cuda 캐시를 비운다.
    """
    if "cuda" not in device:
        return
    try:
        from hayakoe.nlp import bert_models

        bert_models.unload_model()
    except Exception:
        pass
    try:
        import torch

        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _verify_one(
    target_name: str,
    destination_uri: str,
    token: Optional[str],
    device: str,
    out_path: Path,
) -> None:
    """단일 device(``cpu``/``cuda``) 로 다운로드 → 합성 → WAV 저장."""
    from hayakoe import TTS
    from hayakoe.constants import default_cache_dir

    # BERT 같은 공용 리소스가 매 검증마다 재다운로드되지 않도록 persistent
    # 캐시를 쓴다. HF snapshot_download 가 커밋 해시 기반이라 방금 올린
    # 화자 파일은 어차피 새로 받아옴.
    label = "CPU (ONNX)" if device == "cpu" else "GPU (PyTorch)"
    try:
        tts = (
            TTS(device=device, cache_dir=default_cache_dir(), hf_token=token)
            .load(target_name, source=destination_uri)
            .prepare()
        )
        speaker = tts.speakers[target_name]
        audio = speaker.generate(_VERIFY_TEXT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio.save(out_path)
        duration = len(audio.data) / audio.sr
        console.print(
            f"  [success]✓ {label}[/success] [dim]— {duration:.2f}s @ {audio.sr}Hz[/dim]"
        )
        console.print(f"    [dim]→ {out_path}[/dim]")
    except Exception as e:
        console.print(
            f"  [error]✗ {label} 검증 실패[/error] "
            f"[dim]{type(e).__name__}: {e}[/dim]"
        )
    finally:
        _reset_gpu_state_if_needed(device)


def _verify_via_runtime(
    target_name: str,
    destination_uri: str,
    token: Optional[str],
    backends: list[str],
) -> None:
    """업로드 직후 hayakoe 런타임으로 실제 다운로드 + 추론을 돌려 본다.

    - ``onnx`` 백엔드 → CPU 로 검증
    - ``pytorch`` 백엔드 + CUDA 사용 가능 → GPU 로도 검증
    - 합성 결과는 ``dev-tools/.verify_audio/`` 에 WAV 로 저장되어 직접 들어볼 수 있다.
    """
    console.print()
    console.print(
        "  [accent]검증[/accent] "
        f"[dim]— 방금 올린 주소에서 실제로 받아 \"{_VERIFY_TEXT}\" 를 합성합니다.[/dim]"
    )

    # dev-tools/.verify_audio/<target>_<device>.wav
    out_root = Path(__file__).resolve().parents[2] / ".verify_audio"

    if "onnx" in backends:
        _verify_one(
            target_name, destination_uri, token,
            device="cpu",
            out_path=out_root / f"{target_name}_cpu.wav",
        )

    if "pytorch" in backends:
        cuda_ok = False
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except ModuleNotFoundError:
            cuda_ok = False
        if cuda_ok:
            _verify_one(
                target_name, destination_uri, token,
                device="cuda",
                out_path=out_root / f"{target_name}_cuda.wav",
            )
        else:
            console.print(
                "  [dim]CUDA 를 사용할 수 없어 GPU(PyTorch) 검증은 스킵합니다.[/dim]"
            )

    if not backends or ("onnx" not in backends and "pytorch" not in backends):
        console.print("  [dim]검증할 백엔드가 없습니다.[/dim]")


# ──────────────────────────── 기존 데이터 덮어쓰기 ────────────────────────────


def _remote_speaker_dirs_present(
    source,
    target_name: str,
    backends: list[str],
    token: Optional[str],
) -> list[str]:
    """``{backend}/speakers/{target_name}/`` 이미 존재하는 prefix 목록 반환."""
    prefixes = [f"{backend}/speakers/{target_name}" for backend in backends]
    existing: list[str] = []

    if isinstance(source, LocalSource):
        for p in prefixes:
            if (source.root / p).exists():
                existing.append(p)
        return existing

    if isinstance(source, S3Source):
        try:
            import boto3

            client = boto3.client("s3")
        except Exception:
            return existing
        for p in prefixes:
            key_prefix = f"{source.prefix}/{p}/" if source.prefix else f"{p}/"
            try:
                resp = client.list_objects_v2(
                    Bucket=source.bucket, Prefix=key_prefix, MaxKeys=1,
                )
                if resp.get("KeyCount", 0) > 0:
                    existing.append(p)
            except Exception:
                pass
        return existing

    if isinstance(source, HFSource):
        try:
            from huggingface_hub import HfApi

            files = HfApi(token=token or source.token).list_repo_files(
                source.repo, revision=source.revision,
            )
            for p in prefixes:
                needle = p + "/"
                if any(f.startswith(needle) for f in files):
                    existing.append(p)
        except Exception:
            pass
        return existing

    return existing


def _wipe_remote_speaker_dirs(
    source,
    prefixes: list[str],
    token: Optional[str],
) -> None:
    """지정된 prefix 들을 remote 에서 깔끔하게 삭제한다."""
    if not prefixes:
        return

    if isinstance(source, LocalSource):
        for p in prefixes:
            target = source.root / p
            if target.exists():
                shutil.rmtree(target)
            console.print(f"  [dim]wipe (local) → {target}[/dim]")
        return

    if isinstance(source, S3Source):
        import boto3

        client = boto3.client("s3")
        for p in prefixes:
            key_prefix = f"{source.prefix}/{p}/" if source.prefix else f"{p}/"
            paginator = client.get_paginator("list_objects_v2")
            batch: list[dict] = []
            total = 0
            for page in paginator.paginate(Bucket=source.bucket, Prefix=key_prefix):
                for obj in page.get("Contents", []):
                    batch.append({"Key": obj["Key"]})
                    if len(batch) >= 1000:
                        client.delete_objects(
                            Bucket=source.bucket,
                            Delete={"Objects": batch},
                        )
                        total += len(batch)
                        batch = []
            if batch:
                client.delete_objects(
                    Bucket=source.bucket,
                    Delete={"Objects": batch},
                )
                total += len(batch)
            console.print(f"  [dim]wipe (s3) → {key_prefix} ({total} objects)[/dim]")
        return

    if isinstance(source, HFSource):
        from huggingface_hub import HfApi

        api = HfApi(token=token or source.token)
        for p in prefixes:
            try:
                api.delete_folder(
                    path_in_repo=p,
                    repo_id=source.repo,
                    revision=source.revision,
                )
                console.print(f"  [dim]wipe (hf) → {p}/[/dim]")
            except Exception as e:
                console.print(
                    f"  [warning]HF delete_folder({p}) 실패: "
                    f"{type(e).__name__}: {e}[/warning]"
                )
        return


# ──────────────────────────── 메인 플로우 ────────────────────────────


def publish_menu():
    """화자 배포 메인 메뉴."""
    console.print()
    console.print(Panel(
        "[accent]화자 배포 (Publish)[/accent] [dim]— HF / S3 / 로컬[/dim]\n\n"
        "[dim]학습이 끝난 화자를 HuggingFace private repo, S3 버킷, 또는\n"
        "로컬 디렉토리에 올려 런타임에서 다운로드 받아 쓸 수 있게 합니다.\n"
        "이후 런타임에서\n"
        "  TTS(...).load(name, source=\"hf://...\").prepare()\n"
        "로 받아 쓸 수 있습니다.\n\n"
        "CPU 배포를 고르면 필요 시 ONNX 내보내기를 자동으로 수행합니다.[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    datasets = discover_datasets()
    dataset_sources: list[PublishSource] = []
    for ds in datasets or []:
        ps = _publish_source_from_dataset(ds)
        if ps is not None:
            dataset_sources.append(ps)

    # 화자 선택 — 학습된 dataset + "폴더에서 직접 선택" 을 한 메뉴에 섞는다
    _FOLDER_PICK = "📁 다른 폴더에서 직접 선택..."
    choices = [src.display_name for src in dataset_sources] + [_FOLDER_PICK, "뒤로"]
    if not dataset_sources:
        console.print(
            "  [dim]학습된 화자가 없습니다 — 폴더에서 직접 선택해 올릴 수 있습니다.[/dim]"
        )
    name = select_from_list("화자 선택", choices)
    if name == "뒤로":
        return

    if name == _FOLDER_PICK:
        raw = text_input(
            "폴더 경로 (config.json + style_vectors.npy + *.safetensors 포함)",
        ).strip()
        if not raw:
            return
        folder = Path(raw).expanduser().resolve()
        try:
            src = _publish_source_from_folder(folder)
        except ValueError as e:
            console.print(f"  [error]{e}[/error]")
            return
        console.print(f"  [dim]폴더 선택 → {folder}[/dim]")
    else:
        src = next(s for s in dataset_sources if s.display_name == name)

    # 백엔드 선택 (항상 모두 오픈)
    backends = _select_backends()
    if not backends:
        return

    # 체크포인트 선택
    ckpt = _select_checkpoint(src)
    if ckpt is None:
        return

    # 배포 이름 (런타임 load() 인자로 쓰임)
    target_name = edit_value(
        "배포 화자 이름 (런타임 load() 인자)", src.default_target_name,
    )
    if not target_name:
        return

    # 목적지 종류
    dest_kind = select_from_list(
        "어디서 다운로드 받을 수 있게 할까요?",
        [
            "HuggingFace Hub",
            "S3 (또는 S3-호환)",
            "로컬 디렉토리 (이 머신에 보관)",
            "뒤로",
        ],
    )
    if dest_kind == "뒤로":
        return

    if dest_kind == "HuggingFace Hub":
        destination, token = _prompt_hf_destination()
    elif dest_kind.startswith("S3"):
        destination = _prompt_s3_destination()
        token = None
    else:
        destination = _prompt_local_destination()
        token = None

    if not destination:
        return

    # 요약
    console.print()
    console.print("  [accent]배포 요약[/accent]")
    console.print(f"  화자:        [value]{src.display_name}[/value]")
    console.print(f"  배포 이름:   [value]{target_name}[/value]")
    console.print(f"  백엔드:      [value]{', '.join(backends)}[/value]")
    console.print(f"  체크포인트:  [value]{ckpt}[/value]")
    console.print(f"  대상 종류:   [value]{dest_kind}[/value]")
    console.print(f"  대상:        [value]{destination}[/value]")
    console.print()

    if not confirm("배포를 시작하시겠습니까?", default=False):
        return

    # 소스 파싱
    cache_dir = Path.cwd() / "hayakoe_cache"
    source = parse_source(destination, cache_dir=cache_dir, token=token)

    # 목적지에 같은 화자 디렉토리가 이미 있는지 — 있으면 확인 후 깨끗이 지움
    existing_prefixes = _remote_speaker_dirs_present(
        source, target_name, backends, token,
    )
    if existing_prefixes:
        console.print()
        console.print(
            "  [warning]목적지에 해당 화자 디렉토리가 이미 존재합니다:[/warning]"
        )
        for p in existing_prefixes:
            console.print(f"    [dim]- {p}/[/dim]")
        console.print(
            "  [dim]그대로 업로드하면 파일명이 다른 구버전 아티팩트가 섞여 남을 수\n"
            "  있습니다. 진행하면 위 디렉토리들을 완전히 지운 뒤 새 파일로\n"
            "  덮어씁니다 — 다른 화자와 루트 파일(README 등) 은 건드리지 않습니다.[/dim]"
        )
        if not confirm("기존 디렉토리를 지우고 덮어쓸까요?", default=False):
            return

    # ONNX export (필요 시) — overwrite 확정 후 수행 (작업 낭비 방지)
    onnx_dir: Path | None = None
    if "onnx" in backends:
        onnx_dir = _ensure_onnx_exports(src, ckpt)

    # 스테이징 + 업로드
    with tempfile.TemporaryDirectory(prefix="hayakoe-publish-") as tmp:
        staging = Path(tmp)

        if "pytorch" in backends:
            pt_dir = staging / "pytorch" / "speakers" / target_name
            _stage_pytorch(src, ckpt, pt_dir)
            console.print(f"  [dim]stage → {pt_dir.relative_to(staging)}[/dim]")

        if "onnx" in backends:
            assert onnx_dir is not None
            onnx_out = staging / "onnx" / "speakers" / target_name
            _stage_onnx(src, onnx_dir, onnx_out)
            console.print(f"  [dim]stage → {onnx_out.relative_to(staging)}[/dim]")

        # README 자동 생성 (루트에 없을 때 한해, 유저 승인 후)
        _maybe_stage_readme(staging, source, destination, token, target_name)

        # 기존 디렉토리 정리 (upload 직전)
        if existing_prefixes:
            console.print("  [dim]기존 디렉토리 정리 중...[/dim]")
            _wipe_remote_speaker_dirs(source, existing_prefixes, token)

        console.print("  [dim]업로드 중...[/dim]")
        # root 기준으로 한 번에 업로드 — Source 가 재귀 처리
        source.upload(prefix="", local_dir=staging)

    console.print()
    console.print("[success]배포 완료![/success]")
    console.print("  [dim]런타임 사용 예:[/dim]")
    console.print(
        f"  [dim]TTS(...).load(\"{target_name}\", source=\"{destination}\").prepare()[/dim]"
    )

    # 검증 — 실제로 받아서 추론까지
    _verify_via_runtime(target_name, destination, token, backends)
    console.print()
