# python_backend/Dockerfile

FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . .

# 앱 실행
CMD ["uvicorn", "run_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
