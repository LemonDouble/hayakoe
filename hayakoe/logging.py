from loguru import logger


# loguru 의 라이브러리 관례를 따른다: 라이브러리는 전역 logger 의 핸들러를
# 추가/제거하지 않고, 자기 네임스페이스 로그를 기본 비활성화해 둔다.
# (모듈 최상위에서 logger.remove()/add() 를 하면 hayakoe 를 import 하는
# 호스트 애플리케이션이 등록한 sink 가 조용히 전부 삭제된다)
#
# - 라이브러리 사용자: ``from loguru import logger; logger.enable("hayakoe")``
#   로 hayakoe 내부 로그를 켤 수 있다 (loguru 기본 stderr 핸들러로 출력).
# - CLI / 단독 실행 진입점: :func:`setup_logging` 을 호출하면 기존의 보기
#   좋은 포맷으로 활성화된다.
logger.disable("hayakoe")


def setup_logging() -> None:
    """CLI/단독 실행용 로깅 설정.

    hayakoe 로그를 활성화하고 전역 logger 를 hayakoe 포맷 핸들러 하나로
    재구성한다. 전역 상태를 바꾸므로 애플리케이션 진입점에서만 호출할 것
    — 라이브러리 코드에서는 절대 호출하면 안 된다.
    """
    from hayakoe.utils.stdout_wrapper import SAFE_STDOUT

    logger.enable("hayakoe")
    logger.remove()
    logger.add(
        SAFE_STDOUT,
        format="<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}",
        backtrace=True,
        diagnose=True,
    )
