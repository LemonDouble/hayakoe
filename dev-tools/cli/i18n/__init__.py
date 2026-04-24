"""CLI i18n — 언어 감지 + 번역 함수.

우선순위:
  1. HAYAKOE_LANG 환경변수 (ko / ja / zh / en)
  2. ~/.config/hayakoe/config.toml 의 lang 필드
  3. 시스템 로캘 (LANG, LC_ALL 등) 에서 추론
  4. 사용자에게 직접 물어보기 (인터랙티브 모드)
  5. fallback: en
"""

from __future__ import annotations

import json
import locale
import os
import sys
from pathlib import Path
from typing import Any

_SUPPORTED = ("ko", "ja", "zh", "en")
_LANG_LABELS = {"ko": "한국어", "ja": "日本語", "zh": "中文", "en": "English"}
_CONFIG_DIR = Path.home() / ".config" / "hayakoe"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"

_strings: dict[str, str] = {}
_current_lang: str = "en"


def _detect_from_env() -> str | None:
    env = os.environ.get("HAYAKOE_LANG", "").strip().lower()
    if env in _SUPPORTED:
        return env
    return None


def _detect_from_config() -> str | None:
    if not _CONFIG_FILE.exists():
        return None
    try:
        text = _CONFIG_FILE.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("lang"):
                val = line.split("=", 1)[1].strip().strip('"').strip("'").lower()
                if val in _SUPPORTED:
                    return val
    except Exception:
        pass
    return None


def _detect_from_locale() -> str | None:
    sys_locale = os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
    if not sys_locale:
        try:
            sys_locale = locale.getdefaultlocale()[0] or ""
        except Exception:
            sys_locale = ""
    sys_locale = sys_locale.lower()
    if sys_locale.startswith("ko"):
        return "ko"
    if sys_locale.startswith("ja"):
        return "ja"
    if sys_locale.startswith("zh"):
        return "zh"
    if sys_locale.startswith("en"):
        return "en"
    return None


def _ask_user() -> str | None:
    if not sys.stdin.isatty():
        return None
    print()
    print("  Select language / 언어를 선택하세요:")
    print()
    for i, code in enumerate(_SUPPORTED, 1):
        print(f"    {i}. {_LANG_LABELS[code]}")
    print()
    try:
        raw = input("  → (1-4): ").strip()
        idx = int(raw) - 1
        if 0 <= idx < len(_SUPPORTED):
            chosen = _SUPPORTED[idx]
            set_lang(chosen)
            return chosen
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    return None


def _detect_lang() -> str:
    return (
        _detect_from_env()
        or _detect_from_config()
        or _detect_from_locale()
        or _ask_user()
        or "en"
    )


def _load_strings(lang: str) -> dict[str, str]:
    locale_dir = Path(__file__).parent / "locales"
    path = locale_dir / f"{lang}.json"
    if not path.exists():
        path = locale_dir / "ko.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init(lang: str | None = None) -> str:
    """i18n 초기화. 반환값은 감지/선택된 언어 코드."""
    global _strings, _current_lang
    _current_lang = lang if lang and lang in _SUPPORTED else _detect_lang()
    _strings = _load_strings(_current_lang)
    return _current_lang


def get_lang() -> str:
    return _current_lang


def set_lang(lang: str) -> None:
    """언어를 변경하고 설정 파일에 저장."""
    global _strings, _current_lang
    if lang not in _SUPPORTED:
        raise ValueError(f"Unsupported language: {lang}. Use one of {_SUPPORTED}")
    _current_lang = lang
    _strings = _load_strings(lang)
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(f'lang = "{lang}"\n', encoding="utf-8")
    except Exception:
        pass


def t(key: str, **kwargs: Any) -> str:
    """번역된 문자열을 반환한다. kwargs 로 포맷팅."""
    text = _strings.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text
