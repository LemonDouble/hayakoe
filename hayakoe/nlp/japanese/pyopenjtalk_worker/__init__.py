"""
사용자 사전 접근 오류를 방지하기 위해
pyopenjtalk 워커를 별도 프로세스에서 실행
"""

from typing import Any, Optional

from hayakoe.logging import logger
from hayakoe.nlp.japanese.pyopenjtalk_worker.worker_client import WorkerClient
from hayakoe.nlp.japanese.pyopenjtalk_worker.worker_common import WORKER_PORT


WORKER_CLIENT: Optional[WorkerClient] = None


# pyopenjtalk 인터페이스
# g2p(): 사용하지 않음


def run_frontend(text: str) -> list[dict[str, Any]]:
    if WORKER_CLIENT is not None:
        ret = WORKER_CLIENT.dispatch_pyopenjtalk("run_frontend", text)
        assert isinstance(ret, list)
        return ret
    else:
        # 워커 없이 실행
        import pyopenjtalk

        return pyopenjtalk.run_frontend(text)


def make_label(njd_features: Any) -> list[str]:
    if WORKER_CLIENT is not None:
        ret = WORKER_CLIENT.dispatch_pyopenjtalk("make_label", njd_features)
        assert isinstance(ret, list)
        return ret
    else:
        # 워커 없이 실행
        import pyopenjtalk

        return pyopenjtalk.make_label(njd_features)


def mecab_dict_index(path: str, out_path: str, dn_mecab: Optional[str] = None) -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("mecab_dict_index", path, out_path, dn_mecab)
    else:
        # 워커 없이 실행
        import pyopenjtalk

        pyopenjtalk.mecab_dict_index(path, out_path, dn_mecab)


def update_global_jtalk_with_user_dict(path: str) -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("update_global_jtalk_with_user_dict", path)
    else:
        # 워커 없이 실행
        import pyopenjtalk

        pyopenjtalk.update_global_jtalk_with_user_dict(path)


def unset_user_dict() -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("unset_user_dict")
    else:
        # 워커 없이 실행
        import pyopenjtalk

        pyopenjtalk.unset_user_dict()


# 임포트 시 모듈 초기화


def initialize_worker(port: int = WORKER_PORT) -> None:
    import atexit
    import signal
    import socket
    import sys
    import time

    global WORKER_CLIENT
    if WORKER_CLIENT:
        return

    client = None
    try:
        client = WorkerClient(port)
    except (OSError, socket.timeout):
        logger.debug("try starting pyopenjtalk worker server")
        import os
        import subprocess

        worker_pkg_path = os.path.relpath(
            os.path.dirname(__file__), os.getcwd()
        ).replace(os.sep, ".")
        args = [sys.executable, "-m", worker_pkg_path, "--port", str(port)]
        # 새 세션, 새 프로세스 그룹
        if sys.platform.startswith("win"):
            cf = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            si = subprocess.STARTUPINFO()  # type: ignore
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore
            si.wShowWindow = subprocess.SW_HIDE  # type: ignore
            subprocess.Popen(args, creationflags=cf, startupinfo=si)
        else:
            # Windows 동작에 맞춤
            # start_new_session은 preexec_fn에서 setsid를 지정하는 것과 동일
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # 서버가 리스닝할 때까지 대기
        count = 0
        while True:
            try:
                client = WorkerClient(port)
                break
            except OSError:
                time.sleep(0.5)
                count += 1
                # 20: 최대 재시도 횟수
                if count == 20:
                    raise TimeoutError("서버에 연결할 수 없습니다")

    logger.debug("pyopenjtalk worker server started")
    WORKER_CLIENT = client
    atexit.register(terminate_worker)

    # 프로세스가 종료될 때
    def signal_handler(signum: int, frame: Any):
        terminate_worker()

    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # signal은 메인 스레드에서만 동작
        pass


# 최상위 레벨 선언
def terminate_worker() -> None:
    logger.debug("pyopenjtalk worker server terminated")
    global WORKER_CLIENT
    if not WORKER_CLIENT:
        return

    # 예기치 않은 오류에 대비
    try:
        if WORKER_CLIENT.status() == 1:
            WORKER_CLIENT.quit_server()
    except Exception as e:
        logger.error(e)

    WORKER_CLIENT.close()
    WORKER_CLIENT = None
