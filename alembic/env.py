# alembic/env.py
# 1. Alembic 설정 파일을 위한 경로 설정
import sys
from os.path import abspath, dirname

# 프로젝트 루트 폴더를 파이썬 경로에 추가
sys.path.insert(0, dirname(dirname(abspath(__file__))))


# ------------------- 2. Alembic 기본 설정 --------------------
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os

# --- 3. 사용자 모델(Base) 및 데이터베이스 URL 설정 ---
from python_backend.database.models import Base

config = context.config

config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL"))

target_metadata = Base.metadata

# ------------------- 4. Alembic 런타임 설정 ------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
