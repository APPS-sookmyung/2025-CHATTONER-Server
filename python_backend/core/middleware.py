"""
FastAPI 미들웨어 설정
CORS, 로깅, 에러 핸들링 등의 미들웨어 구성
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging


def setup_middleware(app: FastAPI, settings):
    """미들웨어 설정"""
    
    # CORS 미들웨어
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 개발 환경에서는 모든 도메인 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 로깅 미들웨어 (추후 확장 가능)
    if settings.DEBUG:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)