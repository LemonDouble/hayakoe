"""전처리 서버 설정."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    device: str = "cuda"
    whisper_model: str = "large-v3"
    separator_model: str = "mel_band_roformer_kim_ft_unwa.ckpt"


_settings: Settings | None = None


def init(data_dir: str | Path, **kwargs):
    global _settings
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "videos").mkdir(exist_ok=True)
    _settings = Settings(data_dir=data_dir, **kwargs)


def get() -> Settings:
    if _settings is None:
        raise RuntimeError("config.init()을 먼저 호출하세요.")
    return _settings
