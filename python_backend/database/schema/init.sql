-- 통합된 최종 init.sql 스크립트
-- 기본 사용자 기능과 기업용 확장 기능을 모두 포함하며, 데이터 정합성과 일관성을 보장합니다.

BEGIN;

-- =================================================================
-- 1. 개인 사용자 관련 테이블 (init.sql 기반)
-- =================================================================

-- 1-1. 사용자 테이블
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_users_username ON users(username);

-- 1-2. 사용자 프로필 테이블
CREATE TABLE IF NOT EXISTS user_profiles (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    base_formality_level INTEGER DEFAULT 3,
    base_friendliness_level INTEGER DEFAULT 3,
    base_emotion_level INTEGER DEFAULT 3,
    base_directness_level INTEGER DEFAULT 3,
    session_formality_level REAL,
    session_friendliness_level REAL,
    session_emotion_level REAL,
    session_directness_level REAL,
    questionnaire_responses JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_user_profiles_user_id ON user_profiles(user_id);

-- 1-3. 변환 기록 테이블
CREATE TABLE IF NOT EXISTS conversion_history (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    original_text TEXT NOT NULL,
    converted_texts JSONB NOT NULL,
    context VARCHAR(50) DEFAULT 'personal',
    user_rating INTEGER,
    selected_version VARCHAR(20),
    feedback_text TEXT,
    sentiment_analysis JSONB DEFAULT '{}',
    prompts_used JSONB DEFAULT '{}',
    model_used VARCHAR(50) DEFAULT 'gpt-4o',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_conversion_history_user_id ON conversion_history(user_id);

-- 1-4. 네거티브 프롬프트 설정 테이블
CREATE TABLE IF NOT EXISTS negative_preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    avoid_flowery_language VARCHAR(20) DEFAULT 'moderate',
    avoid_repetitive_words VARCHAR(20) DEFAULT 'moderate',
    comma_usage_style VARCHAR(20) DEFAULT 'moderate',
    content_over_format VARCHAR(20) DEFAULT 'moderate',
    bullet_point_usage VARCHAR(20) DEFAULT 'moderate',
    emoticon_usage VARCHAR(20) DEFAULT 'strict',
    custom_negative_prompts JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_negative_preferences_user_id ON negative_preferences(user_id);

-- =================================================================
-- 2. 기업용 확장 테이블 (database_migration.sql 기반, 수정 및 통합)
-- =================================================================

-- 2-1. 기업 프로필 테이블 (migration 파일의 상세한 구조를 채택)
CREATE TABLE IF NOT EXISTS company_profiles (
    id BIGSERIAL PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    team_size INTEGER,
    primary_business TEXT,
    communication_style VARCHAR(50), -- friendly, strict, formal
    main_channels JSONB, -- ["slack", "discord", "email"]
    target_audience JSONB, -- ["internal", "clients", "management"]
    generated_profile TEXT, -- LLM이 생성한 커뮤니케이션 프로필
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_company_profiles_company_name ON company_profiles(company_name);


-- 2-2. 사용자-기업 연결 테이블 (역할 및 상태 관리)
CREATE TABLE IF NOT EXISTS user_company_relations (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    company_id BIGINT NOT NULL REFERENCES company_profiles(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'employee', -- admin, hr, employee
    status VARCHAR(20) DEFAULT 'active', -- active, inactive, pending
    onboarding_completed BOOLEAN DEFAULT FALSE,
    joined_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, company_id)
);
CREATE INDEX IF NOT EXISTS ix_user_company_relations_user_id ON user_company_relations(user_id);
CREATE INDEX IF NOT EXISTS ix_user_company_relations_company_id ON user_company_relations(company_id);


-- 2-3. 커뮤니케이션 가이드라인 테이블
CREATE TABLE IF NOT EXISTS communication_guidelines (
    id BIGSERIAL PRIMARY KEY,
    company_id BIGINT NOT NULL REFERENCES company_profiles(id) ON DELETE CASCADE,
    document_type VARCHAR(50) NOT NULL, -- pdf, text, manual_input
    document_name VARCHAR(255),
    document_content TEXT NOT NULL,
    processed_chunks JSONB, -- 처리된 텍스트 청크
    vector_embeddings JSONB, -- 벡터 임베딩
    chunk_metadata JSONB, -- 청크 메타데이터
    uploaded_by BIGINT REFERENCES users(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_communication_guidelines_company_id ON communication_guidelines(company_id);


-- 2-4. 기업용 사용자 피드백 테이블
CREATE TABLE IF NOT EXISTS company_user_feedback (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    company_id BIGINT NOT NULL REFERENCES company_profiles(id) ON DELETE CASCADE,
    session_id VARCHAR(100),
    original_text TEXT NOT NULL,
    suggested_text TEXT NOT NULL,
    feedback_type VARCHAR(20) NOT NULL, -- grammar, protocol
    feedback_value VARCHAR(10) NOT NULL, -- good, bad
    applied_in_final BOOLEAN DEFAULT FALSE,
    metadata JSONB, -- 컨텍스트 정보
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_company_user_feedback_user_id ON company_user_feedback(user_id);
CREATE INDEX IF NOT EXISTS ix_company_user_feedback_company_id ON company_user_feedback(company_id);


-- 2-5. 기업 학습 데이터 테이블
CREATE TABLE IF NOT EXISTS company_learning_data (
    id BIGSERIAL PRIMARY KEY,
    company_id BIGINT NOT NULL REFERENCES company_profiles(id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    feedback VARCHAR(10), -- good, bad, neutral
    context JSONB, -- 상황, 대상, 채널 정보
    embedding JSONB, -- 벡터 임베딩
    is_approved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_company_learning_data_company_id ON company_learning_data(company_id);


-- 2-6. 사용자별 기업 선호도 테이블
CREATE TABLE IF NOT EXISTS user_company_preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    company_id BIGINT NOT NULL,
    personal_style JSONB, -- 개인 스타일 설정
    negative_prompts TEXT, -- 피하고 싶은 스타일
    preference_weights JSONB, -- 문법/프로토콜/개인 가중치
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (company_id) REFERENCES company_profiles(id) ON DELETE CASCADE,
    UNIQUE(user_id, company_id)
);
CREATE INDEX IF NOT EXISTS ix_user_company_preferences_user_company ON user_company_preferences(user_id, company_id);


-- 2-7. 설문조사 응답 테이블
CREATE TABLE IF NOT EXISTS company_survey_responses (
    id BIGSERIAL PRIMARY KEY,
    company_id BIGINT NOT NULL REFERENCES company_profiles(id) ON DELETE CASCADE,
    respondent_name VARCHAR(255),
    respondent_role VARCHAR(100),
    survey_responses JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_company_survey_responses_company_id ON company_survey_responses(company_id);


COMMIT;