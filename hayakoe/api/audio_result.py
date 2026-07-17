from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray


def streaming_wav_header(
    sample_rate: int, *, channels: int = 1, sampwidth: int = 2
) -> bytes:
    """길이 미정 스트리밍용 WAV 헤더 (44바이트) 를 반환한다.

    RIFF/data 청크 크기를 ``0xFFFFFFFF`` 로 두는 라이브 스트리밍 관례를 따른다.
    총 길이를 미리 알 수 없는 스트림 (HTTP chunked 응답, WebSocket 등) 의
    맨 앞에 한 번만 내보내고, 이후에는 raw PCM (:meth:`AudioResult.pcm_bytes`)
    만 이어서 보내면 된다. ffmpeg / 브라우저 등 대부분의 플레이어가 지원한다.
    """
    byte_rate = sample_rate * channels * sampwidth
    block_align = channels * sampwidth
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 0xFFFFFFFF),
            b"WAVE",
            b"fmt ",
            struct.pack(
                "<IHHIIHH",
                16,  # fmt 청크 크기
                1,  # PCM
                channels,
                sample_rate,
                byte_rate,
                block_align,
                sampwidth * 8,
            ),
            b"data",
            struct.pack("<I", 0xFFFFFFFF),
        ]
    )


@dataclass
class AudioResult:
    """생성된 오디오 데이터."""

    sr: int
    data: NDArray[np.int16]

    def save(self, path: Union[str, Path]) -> None:
        """WAV 파일로 저장한다."""
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sr)
            wf.writeframes(self.data.tobytes())

    def to_bytes(self) -> bytes:
        """완결된 WAV 파일 하나를 bytes로 반환한다 (단일 API 응답용).

        .. warning::
            헤더에 이 청크의 데이터 길이가 기록된 **완결 WAV** 이므로,
            여러 청크의 ``to_bytes()`` 를 이어붙이면 플레이어는 첫 청크만
            재생한다. 스트리밍으로 이어붙일 때는
            :meth:`Speaker.stream_wav` / :meth:`Speaker.astream_wav` 를
            쓰거나, :func:`streaming_wav_header` + :meth:`pcm_bytes` 로
            직접 조립할 것.
        """
        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(self.data.tobytes())
        return buf.getvalue()

    def pcm_bytes(self) -> bytes:
        """헤더 없는 raw PCM (16-bit little-endian mono) bytes를 반환한다.

        :func:`streaming_wav_header` 뒤에 이어붙이는 스트리밍 용도.
        """
        return self.data.tobytes()


class StyleAccessor:
    """스타일 이름에 대한 속성 스타일 접근. IDE 자동완성을 지원한다."""

    def __init__(self, style2id: dict[str, int]) -> None:
        self._style2id = style2id
        for name in style2id:
            object.__setattr__(self, name, name)

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._style2id:
            return name
        available = list(self._style2id.keys())
        raise AttributeError(f"Style '{name}' not found. Available: {available}")

    def __dir__(self) -> list[str]:
        return list(self._style2id.keys()) + list(super().__dir__())

    def __iter__(self):
        return iter(self._style2id.keys())

    def __contains__(self, item: str) -> bool:
        return item in self._style2id

    def __repr__(self) -> str:
        return f"Styles({list(self._style2id.keys())})"
