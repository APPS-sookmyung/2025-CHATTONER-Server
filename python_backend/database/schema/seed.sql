-- seed.sql
-- 개발 및 테스트를 위한 초기 샘플 데이터를 삽입합니다.
-- 주의: 이 스크립트는 데이터베이스가 비어있는 상태에서 실행하는 것을 가정합니다.

BEGIN;

-- 1. 사용자(users) 3명 생성
INSERT INTO users (id, username, password_hash) VALUES
(1, 'admin_user', 'hashed_password_for_admin'),
(2, 'employee_kim', 'hashed_password_for_kim'),
(3, 'employee_lee', 'hashed_password_for_lee');

-- 2. 사용자 프로필(user_profiles) 생성
INSERT INTO user_profiles (user_id, base_formality_level, base_friendliness_level, questionnaire_responses) VALUES
(1, 4, 2, '{"q1": "formal", "q2": "concise"}'),
(2, 3, 4, '{"q1": "friendly", "q2": "detailed"}'),
(3, 3, 3, '{}');

-- 3. 네거티브 프롬프트(negative_preferences) 설정 (employee_kim 사용자)
INSERT INTO negative_preferences (user_id, emoticon_usage, custom_negative_prompts) VALUES
(2, 'strict', '["지나치게 감성적인 표현 피하기", "수동태 문장 사용하지 않기"]');

-- 4. 기업 프로필(company_profiles) 2개 생성
INSERT INTO company_profiles (id, company_name, industry, communication_style, main_channels, target_audience) VALUES
(101, 'FutureTech Inc.', 'IT', 'friendly', '["slack", "email"]', '["internal", "clients"]'),
(102, 'Global Synergy', 'Consulting', 'formal', '["email", "teams"]', '["clients", "management"]');

-- 5. 사용자-기업 관계(user_company_relations) 설정
INSERT INTO user_company_relations (user_id, company_id, role, status, onboarding_completed) VALUES
(1, 101, 'admin', 'active', true),
(2, 101, 'employee', 'active', true),
(3, 102, 'employee', 'pending', false);

-- 6. 커뮤니케이션 가이드라인(communication_guidelines) 추가 (FutureTech Inc.)
INSERT INTO communication_guidelines (company_id, document_type, document_name, document_content, uploaded_by) VALUES
(101, 'manual_input', '고객 응대 이메일 가이드라인', '모든 고객 이메일은 "안녕하세요, [고객명]님. FutureTech의 [담당자명]입니다."로 시작해야 합니다. ...', 1);

-- 7. 사용자별 기업 선호도(user_company_preferences) 설정 (employee_kim)
INSERT INTO user_company_preferences (user_id, company_id, preference_weights) VALUES
(2, 101, '{"grammar": 0.5, "protocol": 0.3, "personal": 0.2}');

-- 8. 기업용 사용자 피드백(company_user_feedback) 생성
INSERT INTO company_user_feedback (user_id, company_id, original_text, suggested_text, feedback_type, feedback_value) VALUES
(2, 101, '수고하세요', '감사합니다. 좋은 하루 보내세요.', 'protocol', 'good');

-- 9. 기업 학습 데이터(company_learning_data) 생성
INSERT INTO company_learning_data (company_id, user_id, input_text, output_text, context, is_approved) VALUES
(101, 1, '회의록 초안 좀 봐주세요.', '팀장님, 지난 회의록 초안 검토를 부탁드립니다.', '{"target": "manager", "channel": "slack"}', true);

-- 10. 설문조사 응답(company_survey_responses) 추가 (FutureTech Inc.)
INSERT INTO company_survey_responses (company_id, respondent_name, respondent_role, survey_responses) VALUES
(101, 'Alice', 'HR Manager', '{"company_culture": "open", "main_tool": "Slack"}');

-- 11. 변환 기록(conversion_history) 생성
INSERT INTO conversion_history (user_id, original_text, converted_texts, context, user_rating, model_used) VALUES
(2, '이거 빨리 좀 처리해주세요.', '{"version1": "이 안건을 신속하게 처리해주시면 감사하겠습니다.", "version2": "안녕하세요, 해당 건에 대해 빠른 검토 부탁드립니다."}', 'personal', 5, 'gpt-4o');

-- 기본 키 시퀀스가 삽입된 값 다음으로 설정되도록 조정합니다.
-- 데이터베이스 환경에 따라 수동 조정이 필요할 수 있습니다.
SELECT setval('users_id_seq', (SELECT MAX(id) FROM users));
SELECT setval('company_profiles_id_seq', (SELECT MAX(id) FROM company_profiles));

COMMIT;