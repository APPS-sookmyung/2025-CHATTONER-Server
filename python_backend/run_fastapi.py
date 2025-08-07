from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 👇 필요 시 CORS 허용 설정 (프론트와 연동 시 필수)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 시 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👋 기본 테스트용 엔드포인트
@app.get("/")
async def root():
    return {"message": "FastAPI 서버가 정상 작동 중입니다 🐾"}

# 👇 여기에 실제 API 라우터를 추가할 수 있습니다
# from api.v1.routes import router as api_router
# app.include_router(api_router, prefix="/api/v1")
