"""HayaKoe 개발 도구 CLI 진입점.

사용법:
    uv run poe cli
"""

import sys
from pathlib import Path

# 프로젝트 루트, dev-tools/, training/core 모듈을 sys.path에 추가
_CLI_DIR = Path(__file__).resolve().parent
_DEV_TOOLS_DIR = _CLI_DIR.parent
_PROJECT_ROOT = _DEV_TOOLS_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_DEV_TOOLS_DIR))
sys.path.insert(0, str(_CLI_DIR / "training" / "core"))

from cli import i18n  # noqa: E402

i18n.init()

from cli.app import app  # noqa: E402


if __name__ == "__main__":
    app()
