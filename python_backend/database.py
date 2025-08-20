# database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 1. os에서 환경 변수(DATABASE_URL)를 불러옵니다.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_toner.db")

# 2. create_engine을 사용해 데이터베이스 엔진을 생성합니다.
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True, 
    connect_args={"check_same_thread": False}
)

# 3. 데이터베이스와 통신할 세션(Session)을 관리하는 클래스를 만듭니다. 
SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)

# 4. SQLAlchemy 모델들이 상속할 기본 클래스를 만듭니다.
Base = declarative_base()

# 5. FastAPI에서 사용할 DB 세션 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()