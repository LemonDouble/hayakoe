# OpenJTalk 사전 번들링

HayaKoe 의 일본어 G2P (발음 변환) 는 [pyopenjtalk](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt) 에 의존합니다.

원본 pyopenjtalk 는 첫 import 시 네트워크에서 사전을 다운로드하는데, HayaKoe 는 **사전을 wheel 안에 번들링한 fork** 를 사용하여 이 지연을 제거했습니다.

## 문제점

원본 pyopenjtalk 는 `import pyopenjtalk` 시점에 OpenJTalk 일본어 사전 (`open_jtalk_dic_utf_8-1.11`, 약 23 MB) 이 로컬에 없으면 **HTTPS 로 자동 다운로드** 합니다.

다운로드 후에는 `~/.local/share/pyopenjtalk/` 에 캐시되어 이후에는 다시 받지 않습니다.

하지만 Docker 컨테이너는 매번 빈 파일시스템으로 시작하므로 **컨테이너를 올릴 때마다 다운로드가 반복** 됩니다.

네트워크가 차단된 환경에서는 import 자체가 실패합니다.

## 구현

자체 fork ([lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)) 에서 **wheel 빌드 시 사전 파일을 패키지 내부에 포함** 하도록 수정했습니다.

- 빌드 워크플로우가 `open_jtalk_dic_utf_8-1.11.tar.gz` 를 다운로드 · 추출하여 `pyopenjtalk/` 패키지 안에 배치
- `pyproject.toml` 의 `package-data`, `MANIFEST.in` 에 사전 포함을 명시

HayaKoe 측은 `pyproject.toml` 의 의존성을 `lemon-pyopenjtalk-prebuilt` 로 지정하는 것으로 적용이 완료됩니다.

설치 후 패키지 내부에 사전이 포함되어 있는 것을 확인할 수 있습니다.

```
site-packages/pyopenjtalk/
  ├─ open_jtalk_dic_utf_8-1.11/   ← 번들링된 사전
  ├─ openjtalk.cpython-310-x86_64-linux-gnu.so
  └─ __init__.py
```

## 개선 효과

- 첫 import 시 네트워크 호출 완전히 제거
- 오프라인 · 폐쇄망 환경에서 설치 직후 즉시 동작
- Docker 이미지 빌드 결과가 네트워크 상태에 무관하게 동일
