"""
api/v1 라우터 설정
- FastAPI 애플리케이션의 API 버전 1 라우터 설정
- 이 파일은 API 엔드포인트를 모듈화하여 관리하기 위한 라우터 설정 
- 각 API 엔드포인트는 별도의 모듈로 분리되어 있으며, 이 파일에서 라우터를 포함

라우터 관련 로직을 한 곳에 모으고 복수의 endpoint 파일을 router.py 에서 한번에 import 및 
등록하게 만듦

주의: 각각 파일 내부에 router= APIRouter() 를 선언해야 함
"""

from fastapi import APIRouter
from api.v1.endpoints.health import router as health_router
from api.v1.endpoints.conversion import router as conversion_router
from api.v1.endpoints.user import router as user_router

api_router=APIRouter()
# v1전체의 라우팅 집합 
# v1 대표 라우터


#health 파일 내 router=APIRouter() 로 선언된 라우터 객체

api_router.include_router(
    health.router,
    prefix="/health", # 이 경로로 시작하는 모든 API 를 이 라우터에 포함 
    tags=["Health Check"] # 스웨거 문서 내에서 health check 라는 그룹으로 분류 
)

api_router.include_router(
    conversion.router,
    prefix="/conversion",
    tags=["Text Conversion"],
    name="text-conversion" # 스웨거 문서 내에서 text conversion 라는 이름으로 분류
)

api_router.include_router(
    user.router,
    prefix="/user",
    tags=["User magagement"]

)