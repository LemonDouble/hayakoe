"""publish 목적지 백엔드 어댑터.

LocalSource / S3Source / HFSource 마다 흩어져 있던 원격 화자 조회 · README
조회 · 존재 확인 · 삭제 로직을 공통 인터페이스로 묶는다::

    dest = make_destination(parse_source(...), token)
    dest.list_speakers() / dest.fetch_readme()
    dest.speaker_dirs_present(prefixes) / dest.wipe(prefixes)

menu 쪽은 isinstance 분기 없이 어댑터 메서드만 호출한다. 백엔드별 예외
처리 규약은 기존 함수들과 동일하게 유지한다 — list_speakers / fetch_readme
는 어떤 실패든 [] / None, speaker_dirs_present 는 S3 에서 prefix 별 개별
삼킴 · HF 에서 전체 삼킴, wipe 는 HF 에서만 prefix 별 에러 출력 후 계속.
"""

import shutil
from pathlib import Path
from typing import Optional, Union

from cli.i18n import t
from cli.ui.console import console

from hayakoe.api.sources import HFSource, LocalSource, S3Source


_BACKENDS = ("pytorch", "onnx")


class LocalDest:
    def __init__(self, source: LocalSource):
        self.source = source

    def list_speakers(self) -> list[str]:
        names: set[str] = set()
        try:
            for backend in _BACKENDS:
                base = self.source.root / backend / "speakers"
                if base.is_dir():
                    for p in base.iterdir():
                        if p.is_dir():
                            names.add(p.name)
        except Exception:
            return []
        return sorted(names)

    def fetch_readme(self) -> Optional[str]:
        try:
            p = self.source.root / "README.md"
            return p.read_text(encoding="utf-8") if p.exists() else None
        except Exception:
            return None

    def speaker_dirs_present(self, prefixes: list[str]) -> list[str]:
        return [p for p in prefixes if (self.source.root / p).exists()]

    def wipe(self, prefixes: list[str]) -> None:
        for p in prefixes:
            target = self.source.root / p
            if target.exists():
                shutil.rmtree(target)
            console.print(f"  [dim]wipe (local) → {target}[/dim]")


class S3Dest:
    def __init__(self, source: S3Source):
        self.source = source

    def list_speakers(self) -> list[str]:
        names: set[str] = set()
        try:
            client = self.source._client()
            for backend in _BACKENDS:
                key_prefix = self.source._key_prefix(f"{backend}/speakers")
                paginator = client.get_paginator("list_objects_v2")
                for page in paginator.paginate(
                    Bucket=self.source.bucket,
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

    def fetch_readme(self) -> Optional[str]:
        try:
            client = self.source._client()
            # _key_prefix("") 는 빈 prefix 에서 "/" 가 되므로 루트 키는 수동 조립
            key = f"{self.source.prefix}/README.md" if self.source.prefix else "README.md"
            obj = client.get_object(Bucket=self.source.bucket, Key=key)
            return obj["Body"].read().decode("utf-8")
        except Exception:
            return None

    def speaker_dirs_present(self, prefixes: list[str]) -> list[str]:
        existing: list[str] = []
        try:
            client = self.source._client()
        except Exception:
            return existing
        for p in prefixes:
            try:
                resp = client.list_objects_v2(
                    Bucket=self.source.bucket,
                    Prefix=self.source._key_prefix(p),
                    MaxKeys=1,
                )
                if resp.get("KeyCount", 0) > 0:
                    existing.append(p)
            except Exception:
                pass
        return existing

    def wipe(self, prefixes: list[str]) -> None:
        client = self.source._client()
        for p in prefixes:
            key_prefix = self.source._key_prefix(p)
            paginator = client.get_paginator("list_objects_v2")
            batch: list[dict] = []
            total = 0
            for page in paginator.paginate(Bucket=self.source.bucket, Prefix=key_prefix):
                for obj in page.get("Contents", []):
                    batch.append({"Key": obj["Key"]})
                    if len(batch) >= 1000:
                        client.delete_objects(
                            Bucket=self.source.bucket,
                            Delete={"Objects": batch},
                        )
                        total += len(batch)
                        batch = []
            if batch:
                client.delete_objects(
                    Bucket=self.source.bucket,
                    Delete={"Objects": batch},
                )
                total += len(batch)
            console.print(f"  [dim]wipe (s3) → {key_prefix} ({total} objects)[/dim]")


class HFDest:
    def __init__(self, source: HFSource, token: Optional[str]):
        self.source = source
        self.token = token or source.token

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def list_speakers(self) -> list[str]:
        names: set[str] = set()
        try:
            files = self._api().list_repo_files(
                self.source.repo, revision=self.source.revision,
            )
            for f in files:
                for backend in _BACKENDS:
                    prefix = f"{backend}/speakers/"
                    if f.startswith(prefix):
                        tail = f[len(prefix):]
                        name = tail.split("/", 1)[0]
                        if name:
                            names.add(name)
        except Exception:
            return []
        return sorted(names)

    def fetch_readme(self) -> Optional[str]:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                self.source.repo,
                "README.md",
                revision=self.source.revision,
                token=self.token,
            )
            return Path(local).read_text(encoding="utf-8")
        except Exception:
            return None

    def speaker_dirs_present(self, prefixes: list[str]) -> list[str]:
        existing: list[str] = []
        try:
            files = self._api().list_repo_files(
                self.source.repo, revision=self.source.revision,
            )
            for p in prefixes:
                needle = p + "/"
                if any(f.startswith(needle) for f in files):
                    existing.append(p)
        except Exception:
            pass
        return existing

    def wipe(self, prefixes: list[str]) -> None:
        api = self._api()
        for p in prefixes:
            try:
                api.delete_folder(
                    path_in_repo=p,
                    repo_id=self.source.repo,
                    revision=self.source.revision,
                )
                console.print(f"  [dim]wipe (hf) → {p}/[/dim]")
            except Exception as e:
                console.print(t("publish.hf.delete_failed", path=p, error_type=type(e).__name__, error=e))


Destination = Union[LocalDest, S3Dest, HFDest]


def make_destination(source, token: Optional[str]) -> Destination:
    """parse_source() 결과를 어댑터로 감싼다."""
    if isinstance(source, LocalSource):
        return LocalDest(source)
    if isinstance(source, S3Source):
        return S3Dest(source)
    if isinstance(source, HFSource):
        return HFDest(source, token)
    raise TypeError(f"지원하지 않는 source 타입: {type(source).__name__}")
