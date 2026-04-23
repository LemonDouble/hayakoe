# Source 추상화 (HF · S3 · 로컬)

화자 모델과 BERT 파일이 어디에 위치하든 **URI 만 바꾸면 동일한 API 로 로드** 할 수 있도록 추상화한 계층입니다.

## 왜 필요한가

화자 모델을 불러오는 소스는 상황에 따라 다릅니다.

- 공개 기본 화자는 **HuggingFace 레포** (`hf://lemondouble/hayakoe`)
- 직접 학습한 화자는 **private HF 레포 · S3 · 로컬 디렉토리** 등

소스마다 다운로드 코드를 분기 처리하면 엔진 본체가 비대해지고, 캐시 경로가 중복되는 문제가 생깁니다.

## 구현

### Source 인터페이스

모든 소스는 **"prefix 단위로 파일을 로컬 캐시에 내려받고 경로를 반환한다"** 는 공통 인터페이스를 구현합니다.

```python
class Source(Protocol):
    def fetch(self, prefix: str) -> Path:
        """prefix/ 아래 모든 파일을 캐시에 받고 로컬 경로를 반환."""
        ...

    def upload(self, prefix: str, local_dir: Path) -> None:
        """local_dir 내용을 prefix/ 아래로 업로드 (배포용)."""
        ...
```

`fetch()` 는 모델 로드 시, `upload()` 는 CLI 의 `publish` (모델 배포) 시 사용됩니다.

### 구현체

| URI 스킴 | 구현 | 동작 |
|---|---|---|
| `hf://user/repo[@revision]` | `HFSource` | `huggingface_hub.snapshot_download()` 로 다운로드. `HF_TOKEN` 환경변수 또는 `hf_token` 파라미터로 private 레포 접근 가능 |
| `s3://bucket/prefix` | `S3Source` | `boto3` 기반. `AWS_ENDPOINT_URL_S3` 환경변수로 S3 호환 엔드포인트 (R2 · MinIO 등) 지원 |
| `file:///abs/path` 또는 `/abs/path` | `LocalSource` | 로컬 디렉토리를 그대로 사용. 다운로드 없음 |

### URI 자동 라우팅

`TTS().load()` 에 URI 만 전달하면, 스킴에 해당하는 Source 가 자동 선택됩니다.

```python
# HuggingFace (기본값)
tts.load("jvnv-F1-jp")

# HuggingFace — private 레포
tts.load("jvnv-F1-jp", source="hf://myorg/my-voices")

# S3
tts.load("jvnv-F1-jp", source="s3://my-bucket/voices")

# 로컬
tts.load("jvnv-F1-jp", source="/data/models")
```

HuggingFace 웹 URL (`https://huggingface.co/user/repo`) 도 자동으로 `hf://` 형태로 정규화하여 받아들입니다.

### 캐시

모든 소스는 동일한 캐시 루트 하위에 저장됩니다.

캐시 경로는 `HAYAKOE_CACHE` 환경변수로 지정하거나, 미지정 시 `$CWD/hayakoe_cache` 가 기본값입니다.

캐시 정책은 단순합니다 — 파일이 있으면 재사용, 없으면 새로 다운로드합니다.

### BERT 소스 분리

화자 모델과 BERT 모델의 소스를 **별도로 지정** 할 수 있습니다.

```python
TTS(
    device="cpu",
    bert_source="hf://lemondouble/hayakoe",  # BERT 는 공식 레포에서
).load(
    "custom-speaker",
    source="/data/my-models",                 # 화자는 로컬에서
).prepare()
```

기본값은 둘 다 `hf://lemondouble/hayakoe` 입니다.

## 개선 효과

- 엔진 본체에서 스토리지별 분기 코드가 제거되었습니다.
- 신규 스토리지를 추가하려면 `Source` 프로토콜을 구현하는 클래스 하나만 작성하면 됩니다.
- CLI 의 `publish` 명령도 동일한 추상화를 역방향 (`upload`) 으로 사용합니다.
