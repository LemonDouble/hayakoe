"""audio-separator 배경음 제거."""

import asyncio
import shutil
from pathlib import Path

from loguru import logger

import config

_lock = asyncio.Lock()


def is_busy() -> bool:
    """배경음 제거 작업이 실행 중인지 확인."""
    return _lock.locked()


async def separate_vocals(input_path: Path, output_path: Path) -> Path:
    """배경음 제거 → vocals WAV를 output_path에 저장."""
    if _lock.locked():
        raise RuntimeError("배경음 제거가 이미 실행 중입니다. 현재 작업이 끝난 후 다시 시도하세요.")

    async with _lock:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _run():
            import gc
            import tempfile

            import torch
            from audio_separator.separator import Separator

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    sep = Separator(
                        model_file_dir=str(Path.home() / ".cache" / "audio-separator"),
                        output_dir=tmp_dir,
                        output_format="WAV",
                    )
                    sep.load_model(config.get().separator_model)
                    output_files = sep.separate(str(input_path))

                    # vocals 파일 찾기
                    vocals_src = None
                    for f in output_files:
                        p = Path(tmp_dir) / Path(f).name if not Path(f).is_absolute() else Path(f)
                        if "vocal" in p.stem.lower():
                            vocals_src = p
                            break
                    if vocals_src is None and output_files:
                        f = output_files[0]
                        vocals_src = Path(tmp_dir) / Path(f).name if not Path(f).is_absolute() else Path(f)

                    if vocals_src is None:
                        raise RuntimeError("separator 출력 파일을 찾을 수 없습니다.")

                    shutil.copy2(vocals_src, output_path)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"separator: {input_path.name}")
        await asyncio.to_thread(_run)
        logger.info(f"separator 완료: {output_path.name}")
        return output_path
