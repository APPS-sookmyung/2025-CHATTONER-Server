# migrations/env.py
from __future__ import annotations
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ──────────────────────────────────────────────────────────────────────────────
# Alembic Config & Logging
# ──────────────────────────────────────────────────────────────────────────────
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ──────────────────────────────────────────────────────────────────────────────
# Import your models & set target_metadata
#   - 아래 경로를 프로젝트 구조에 맞게 고쳐주세요.
#   - models 모듈 import 자체가 테이블들을 메타데이터에 등록하는 side-effect를 유발합니다.
# ──────────────────────────────────────────────────────────────────────────────
# 예: from python_backend.app import models  # noqa: F401
# 예: from python_backend.app.models import Base
from python_backend.database import models  # noqa: F401  # 모델들을 로드(사이드이펙트)
from python_backend.database.models import Base

# Alembic이 비교에 사용할 메타데이터
target_metadata = Base.metadata

# ──────────────────────────────────────────────────────────────────────────────
# Database URL 주입
#   - alembic.ini의 sqlalchemy.url은 비워두고, 여기서 환경변수로 주입
# ──────────────────────────────────────────────────────────────────────────────
db_url = os.getenv("DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# 선택) 스키마/버전 테이블 등 추가 설정
VERSION_TABLE = os.getenv("ALEMBIC_VERSION_TABLE", "alembic_version")
VERSION_TABLE_SCHEMA = os.getenv("ALEMBIC_VERSION_SCHEMA")  # 예: "public" (없으면 None)

# ──────────────────────────────────────────────────────────────────────────────
# Autogenerate 커스텀 (선택)
#   - 인덱스/뷰/트리거 제외, 특정 테이블 제외 등 필터가 필요하면 조정
# ──────────────────────────────────────────────────────────────────────────────
def include_object(object, name, type_, reflected, compare_to):
    # 예) 뷰는 제외
    if type_ == "table" and getattr(object, "is_view", False):
        return False
    # 예) 특정 테이블 제외
    # if type_ == "table" and name in {"spatial_ref_sys"}:
    #     return False
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Offline mode
# ──────────────────────────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,               # 컬럼 타입 변경 감지
        compare_server_default=True,     # server_default 변경 감지
        include_object=include_object,
        version_table=VERSION_TABLE,
        version_table_schema=VERSION_TABLE_SCHEMA,
        include_schemas=True,            # 여러 스키마를 쓸 때
    )

    with context.begin_transaction():
        context.run_migrations()

# ──────────────────────────────────────────────────────────────────────────────
# Online mode
# ──────────────────────────────────────────────────────────────────────────────
def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        # Alembic에서는 일반적으로 NullPool 사용(마이그 때만 잠깐 연결)
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
            version_table=VERSION_TABLE,
            version_table_schema=VERSION_TABLE_SCHEMA,
            include_schemas=True,
            render_as_batch=False,  # SQLite 등 배치 모드가 필요하면 True
        )

        with context.begin_transaction():
            context.run_migrations()

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
