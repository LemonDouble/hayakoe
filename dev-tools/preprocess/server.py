#!/usr/bin/env python
"""HayaKoe 전처리 서버.

Usage:
    python dev-tools/preprocess/server.py [--data-dir DIR] [--port PORT] [--host HOST]
"""

import argparse
import sys
from pathlib import Path

# 이 파일이 있는 디렉토리를 sys.path에 추가 (패키지 내부 import용)
_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from api import classification, dataset, media, review, speakers, videos


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app(data_dir: str | Path = "./data") -> FastAPI:
    config.init(data_dir)

    app = FastAPI(title="HayaKoe Preprocess", version="0.2.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(speakers.router, prefix="/api")
    app.include_router(videos.router, prefix="/api")
    app.include_router(classification.router, prefix="/api")
    app.include_router(review.router, prefix="/api")
    app.include_router(dataset.router, prefix="/api")
    app.include_router(media.router, prefix="/api")

    @app.get("/api/info")
    def info():
        return {"status": "ok", "data_dir": str(config.get().data_dir)}

    # 프론트엔드 빌드 서빙 (SPA fallback 포함)
    static_dir = _THIS_DIR / "static"
    if static_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

        @app.get("/{path:path}")
        async def spa_fallback(path: str):
            # 정적 파일이 있으면 그것을 반환, 없으면 index.html (SPA 라우팅)
            file = static_dir / path
            if file.is_file():
                return FileResponse(file)
            return FileResponse(static_dir / "index.html")

    return app


def main():
    parser = argparse.ArgumentParser(description="HayaKoe 전처리 서버")
    parser.add_argument("--data-dir", default="./data", help="작업 디렉토리 (default: ./data)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    import uvicorn

    config.init(args.data_dir)
    app = create_app(args.data_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
