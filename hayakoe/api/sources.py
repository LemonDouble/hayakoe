"""원격 소스(HF / S3 / 로컬) 추상화.

모든 소스는 "prefix 단위로 파일을 동기화" 하는 얇은 인터페이스를 공유한다.
- ``fetch(prefix)`` — prefix 아래 모든 파일을 로컬 캐시로 받고 로컬 경로를 반환
- ``upload(prefix, local_dir)`` — local_dir 내용을 prefix 아래로 업로드 (발행용)

URI 스킴 / 허용 형식:
    hf://user/repo[@revision]                  → HuggingFace repo
    https://huggingface.co/user/repo           → hf:// 로 정규화
    https://huggingface.co/user/repo/tree/rev  → hf://user/repo@rev
    s3://bucket/prefix                         → S3 (boto3, AWS_ENDPOINT_URL_S3 env 존중)
    file:///abs/path 또는 /abs/path             → 로컬 디렉터리
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


class Source(Protocol):
    """소스의 최소 인터페이스."""

    def fetch(self, prefix: str) -> Path:
        """``prefix/`` 아래 모든 파일을 캐시에 받고 로컬 경로를 반환."""
        ...

    def upload(self, prefix: str, local_dir: Path) -> None:
        """``local_dir`` 내용을 ``prefix/`` 아래로 업로드한다 (발행용)."""
        ...


# ─────────────────────────── HuggingFace ───────────────────────────


@dataclass
class HFSource:
    repo: str
    cache_dir: Path
    revision: str = "main"
    token: Optional[str] = None

    def fetch(self, prefix: str) -> Path:
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            self.repo,
            allow_patterns=[f"{prefix}/*"],
            cache_dir=str(self.cache_dir / "hf"),
            revision=self.revision,
            token=self.token,
        )
        return Path(local) / prefix

    def upload(self, prefix: str, local_dir: Path) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=prefix,
            repo_id=self.repo,
            revision=self.revision,
        )


# ─────────────────────────── S3 / S3-호환 ───────────────────────────


@dataclass
class S3Source:
    bucket: str
    prefix: str
    cache_dir: Path

    def _client(self):
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "s3:// 스킴을 사용하려면 boto3가 필요합니다.\n"
                "  pip install 'hayakoe[s3]'\n"
                "S3-호환 엔드포인트는 AWS_ENDPOINT_URL_S3 환경변수로 설정하세요."
            ) from e
        return boto3.client("s3")

    def _key_prefix(self, rel: str) -> str:
        parts = [p for p in [self.prefix, rel] if p]
        return "/".join(parts).rstrip("/") + "/"

    def _local_base(self, rel: str) -> Path:
        return self.cache_dir / "s3" / self.bucket / (self.prefix or "_") / rel

    def fetch(self, prefix: str) -> Path:
        client = self._client()
        key_prefix = self._key_prefix(prefix)
        local_base = self._local_base(prefix)
        local_base.mkdir(parents=True, exist_ok=True)

        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                rel = key[len(key_prefix):]
                if not rel or rel.endswith("/"):
                    continue
                dest = local_base / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    continue  # 단순 캐시: 한 번 받으면 재사용
                client.download_file(self.bucket, key, str(dest))
        return local_base

    def upload(self, prefix: str, local_dir: Path) -> None:
        client = self._client()
        key_prefix = self._key_prefix(prefix)
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            key = f"{key_prefix}{rel}"
            client.upload_file(str(path), self.bucket, key)


# ─────────────────────────── 로컬 디렉터리 ───────────────────────────


@dataclass
class LocalSource:
    root: Path
    cache_dir: Path  # 사용 안 함, 인터페이스 일관성 위해 유지

    def fetch(self, prefix: str) -> Path:
        path = self.root / prefix
        if not path.exists():
            raise FileNotFoundError(f"Local source missing: {path}")
        return path

    def upload(self, prefix: str, local_dir: Path) -> None:
        dest = self.root / prefix
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


# ─────────────────────────── URI 파서 ───────────────────────────


_HF_WEB_PREFIX = "https://huggingface.co/"


def normalize_hf_uri(value: str) -> Optional[str]:
    """HF 주소를 ``hf://user/repo[@revision]`` 형태로 정규화한다.

    허용 입력:
        - ``hf://user/repo[@revision]`` (그대로 반환)
        - ``https://huggingface.co/user/repo``
        - ``https://huggingface.co/user/repo/tree/<revision>``
        - ``user/repo`` (짧은 형태 — 그대로 hf:// 접두 붙임)

    HF 로 해석 불가능하면 ``None``.
    """
    value = value.strip().rstrip("/")
    if not value:
        return None

    if value.startswith("hf://"):
        return value

    if value.startswith(_HF_WEB_PREFIX):
        rest = value[len(_HF_WEB_PREFIX):]
        # /tree/<rev> 꼬리 제거 및 revision 추출
        if "/tree/" in rest:
            repo_part, _, tail = rest.partition("/tree/")
            revision = tail.split("/", 1)[0]
            if repo_part.count("/") != 1 or not revision:
                return None
            return f"hf://{repo_part}@{revision}"
        # /blob/.., /resolve/.. 등은 repo 루트만 추출
        for marker in ("/blob/", "/resolve/", "/commit/"):
            if marker in rest:
                rest = rest.split(marker, 1)[0]
                break
        if rest.count("/") != 1:
            return None
        return f"hf://{rest}"

    # 짧은 "user/repo" 형태 (스킴 없음) — ``/`` 한 개 포함 & 절대경로 아님
    if (
        "/" in value
        and not value.startswith("/")
        and not value.startswith(".")
        and ":" not in value
        and value.count("/") == 1
    ):
        return f"hf://{value}"

    return None


def parse_source(
    uri: str,
    cache_dir: Path,
    token: Optional[str] = None,
) -> Source:
    """URI 문자열을 Source 어댑터로 변환한다."""
    # HF 웹 URL 도 받아들인다 — hf:// 로 정규화
    if uri.startswith(_HF_WEB_PREFIX):
        normalized = normalize_hf_uri(uri)
        if normalized is None:
            raise ValueError(f"HuggingFace URL 을 해석할 수 없습니다: {uri}")
        uri = normalized

    if uri.startswith("hf://"):
        rest = uri[len("hf://"):]
        if "@" in rest:
            repo, revision = rest.rsplit("@", 1)
        else:
            repo, revision = rest, "main"
        return HFSource(
            repo=repo,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )

    if uri.startswith("s3://"):
        rest = uri[len("s3://"):]
        if "/" in rest:
            bucket, s3_prefix = rest.split("/", 1)
        else:
            bucket, s3_prefix = rest, ""
        return S3Source(
            bucket=bucket,
            prefix=s3_prefix.rstrip("/"),
            cache_dir=cache_dir,
        )

    if uri.startswith("file://"):
        return LocalSource(root=Path(uri[len("file://"):]), cache_dir=cache_dir)

    # 스킴 없는 경로: 로컬로 해석
    return LocalSource(root=Path(uri), cache_dir=cache_dir)
