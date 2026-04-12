"""배포 시 필요한 자격 증명 확보 + .env 영구 저장.

dev-tools 루트의 ``.env`` 를 단순 ``KEY="VALUE"`` 포맷으로 읽고 쓴다.
python-dotenv 의존을 추가하지 않기 위해 자체 미니 파서를 사용한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cli.ui.console import console
from cli.ui.prompts import confirm, secret_input, text_input


def _env_path() -> Path:
    """``dev-tools/.env`` 절대 경로."""
    # credentials.py → publish/ → cli/ → dev-tools/
    return Path(__file__).resolve().parents[2] / ".env"


def load_env_file() -> None:
    """``.env`` 파일의 키-값을 ``os.environ`` 에 로드한다 (이미 설정된 키는 유지)."""
    path = _env_path()
    if not path.exists():
        return
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _write_env_var(key: str, value: str) -> None:
    """``.env`` 에 KEY 를 upsert 하고 ``os.environ`` 도 갱신한다."""
    path = _env_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    new_line = f'{key}="{value}"'
    updated = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = new_line
            updated = True
            break
    if not updated:
        lines.append(new_line)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    os.environ[key] = value


def _maybe_persist(prompt: str, items: dict[str, str]) -> None:
    """한 번의 confirm 으로 여러 env var 를 .env 에 저장한다."""
    if confirm(prompt, default=True):
        for key, value in items.items():
            _write_env_var(key, value)
        console.print(f"  [dim]→ {_env_path()} 에 저장되었습니다 (chmod 600).[/dim]")
    else:
        # 세션 한정: os.environ 만 갱신
        for key, value in items.items():
            os.environ[key] = value
        console.print("  [dim]→ 이번 세션에만 적용됩니다.[/dim]")


# ──────────────────────────── HuggingFace ────────────────────────────


def ensure_hf_token() -> Optional[str]:
    """HF 업로드에 필요한 token 을 확보한다.

    우선순위: os.environ → .env → 인터랙티브 입력.
    입력 후 승인하면 ``.env`` 에 ``HF_TOKEN`` 으로 저장한다.
    """
    load_env_file()
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            return token

    console.print()
    console.print(
        "  [warning]HuggingFace 토큰이 설정되어 있지 않습니다.[/warning]"
    )
    console.print(
        "  [dim]https://huggingface.co/settings/tokens 에서 "
        "write 권한 토큰을 생성하세요.[/dim]"
    )
    console.print()

    token = secret_input("HF_TOKEN").strip()
    if not token:
        console.print("  [error]토큰이 입력되지 않아 업로드를 중단합니다.[/error]")
        return None

    _maybe_persist(
        "이 토큰을 .env 에 저장할까요?",
        {"HF_TOKEN": token},
    )
    return token


# ──────────────────────────── S3 / S3-호환 ────────────────────────────


def ensure_s3_credentials() -> bool:
    """S3 업로드에 필요한 자격 증명을 확보한다.

    필수: ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``
    선택: ``AWS_ENDPOINT_URL_S3`` (MinIO/R2 등), ``AWS_REGION``

    이미 환경에 존재하면 건드리지 않는다. 반환값은 성공 여부.
    """
    load_env_file()

    have_key = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
    have_secret = bool(os.environ.get("AWS_SECRET_ACCESS_KEY"))

    if have_key and have_secret:
        return True

    console.print()
    console.print("  [warning]AWS 자격 증명이 설정되어 있지 않습니다.[/warning]")
    console.print(
        "  [dim]MinIO / R2 / Tigris 등 S3-호환 엔드포인트를 쓰려면\n"
        "    Endpoint URL 도 함께 입력하세요 (AWS_ENDPOINT_URL_S3).[/dim]"
    )
    console.print()

    access_key = text_input("AWS_ACCESS_KEY_ID").strip()
    if not access_key:
        console.print("  [error]Access Key 가 입력되지 않았습니다.[/error]")
        return False

    secret_key = secret_input("AWS_SECRET_ACCESS_KEY").strip()
    if not secret_key:
        console.print("  [error]Secret Key 가 입력되지 않았습니다.[/error]")
        return False

    endpoint = text_input(
        "Endpoint URL (MinIO/R2 등, 일반 AWS 면 Enter)",
        default=os.environ.get("AWS_ENDPOINT_URL_S3", ""),
    ).strip()
    region = text_input(
        "Region (선택)",
        default=os.environ.get("AWS_REGION", ""),
    ).strip()

    items: dict[str, str] = {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
    }
    if endpoint:
        items["AWS_ENDPOINT_URL_S3"] = endpoint
    if region:
        items["AWS_REGION"] = region

    _maybe_persist(
        "자격 증명을 .env 에 저장할까요?",
        items,
    )
    return True
