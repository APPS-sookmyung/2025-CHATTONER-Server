-- smoke_test.sql
-- 데이터베이스 스키마와 기본 CRUD 작업이 정상적으로 동작하는지 검증합니다.

BEGIN;

-- [Test 1: CREATE] 새로운 임시 사용자 및 회사 생성 테스트
INSERT INTO users (id, username, password_hash) VALUES (999, 'smoke_tester', 'test_password') RETURNING id;
INSERT INTO company_profiles (id, company_name) VALUES (9999, 'Smoke Test Corp') RETURNING id;

-- [Test 2: READ] 데이터 조회 테스트
-- 2-1. 방금 생성한 사용자 조회
SELECT * FROM users WHERE username = 'smoke_tester';

-- 2-2. JOIN을 통한 복합 조회 (FutureTech Inc. 소속 직원 목록)
SELECT
    u.username,
    cp.company_name,
    ucr.role
FROM users u
JOIN user_company_relations ucr ON u.id = ucr.user_id
JOIN company_profiles cp ON cp.id = ucr.company_id
WHERE cp.company_name = 'FutureTech Inc.';

-- [Test 3: UPDATE] 데이터 수정 테스트
-- 'employee_lee'의 상태를 'pending'에서 'active'로 변경
UPDATE user_company_relations
SET status = 'active', onboarding_completed = true
WHERE user_id = 3;

-- 3-1. 수정된 내용 확인
SELECT status, onboarding_completed FROM user_company_relations WHERE user_id = 3;

-- [Test 4: DELETE] 데이터 삭제 및 연쇄 삭제(ON DELETE CASCADE) 테스트
-- 'admin_user' (id=1) 삭제
DELETE FROM users WHERE id = 1;

-- 4-1. users 테이블에서 삭제되었는지 확인 (결과가 없어야 함)
SELECT * FROM users WHERE id = 1;

-- 4-2. ON DELETE CASCADE: user_profiles 에서도 자동으로 삭제되었는지 확인 (결과가 없어야 함)
SELECT * FROM user_profiles WHERE user_id = 1;

-- 4-3. ON DELETE CASCADE: user_company_relations 에서도 자동으로 삭제되었는지 확인 (결과가 없어야 함)
SELECT * FROM user_company_relations WHERE user_id = 1;

-- [Test 5: ON DELETE SET NULL] 연쇄 NULL 설정 테스트
DELETE FROM users WHERE id = 2;

-- 5-1. conversion_history의 user_id가 NULL로 설정되었는지 확인
SELECT * FROM conversion_history WHERE user_id IS NULL;


-- 모든 테스트가 끝나면 모든 변경사항을 롤백하여 데이터베이스를 원래 상태로 되돌립니다.
ROLLBACK;