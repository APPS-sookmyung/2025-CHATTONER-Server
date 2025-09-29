from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 중요: 프록시를 통해 접속하므로 주소는 127.0.0.1:5433 입니다.
DATABASE_URL = "postgresql+psycopg://chattoner-user:chattoner1234U*@127.0.0.1:5433/chattoner-db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()