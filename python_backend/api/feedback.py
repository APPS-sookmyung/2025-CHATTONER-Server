from fastapi import APIRouter, Request, status
from schemas.feedback import UserFeedback
from typing import Dict

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"]
)

@router.post("/", status_code=status.HTTP_201_CREATED)
def submit_and_learn_feedback(request: Request, feedback_data: UserFeedback) -> Dict[str, str]:
    """
    사용자로부터 피드백을 받아 세션에 저장
    """
    session = request.session

    # 세션에 사용자 선호도(preferences) 정보가 없으면 초기화
    if 'preferences' not in session:
        session['preferences'] = {
            "history": [],
            "learned_style": {
                "preferred_transformation_counts": {}
            }
        }

    # 받은 피드백을 기록(history)에 저장
    session['preferences']['history'].append(feedback_data.dict())

    # 선호 말투 학습 로직 (사용자 피드백 연결)
    # 가장 선호하는 변환 결과의 카운트를 1 증가시켜 어떤 스타일을 선호하는지 학습
    pref_trans = feedback_data.preferred_transformation
    counts = session['preferences']['learned_style']['preferred_transformation_counts']
    counts[pref_trans] = counts.get(pref_trans, 0) + 1
    
    # 네거티브 프롬프트 저장 (필요 시)
    if feedback_data.negative_prompts:
        if 'negative_prompts' not in session['preferences']['learned_style']:
            session['preferences']['learned_style']['negative_prompts'] = []
        # 중복을 제거하고 추가
        existing_prompts = set(session['preferences']['learned_style']['negative_prompts'])
        existing_prompts.update(feedback_data.negative_prompts)
        session['preferences']['learned_style']['negative_prompts'] = list(existing_prompts)


    # 변경된 세션 데이터는 FastAPI가 자동으로 저장
    return {"message": "피드백이 성공적으로 반영되었습니다."}


@router.get("/preferences", response_model=Dict)
def get_user_preferences(request: Request):
    """
    현재 세션에 저장된 사용자 선호도 데이터를 반환합니다.
    (개발 및 말투 변환 적용 시 활용)
    """
    # 세션에 선호도 데이터가 없으면 빈 객체를 반환
    return request.session.get('preferences', {})