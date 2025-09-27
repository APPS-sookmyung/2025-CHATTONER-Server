"""
Text Quality Analysis Endpoints (RAG-based)
Text quality analysis and suggestion endpoint using RAG
"""

from fastapi import APIRouter, HTTPException, Depends
import json
import re
import statistics
from typing import Annotated, Dict, List, Any

# Changed dependency to use RAG service
from services.rag_service import RAGService
from api.v1.schemas.quality import (
    QualityAnalysisRequest,
    QualityAnalysisResponse,
    ContextSuggestionsRequest,
    ContextSuggestionsResponse,
    SuggestionItem
)
import logging

logger = logging.getLogger('chattoner')
router = APIRouter()

# RAG service dependency injection function
def get_rag_service():
    """Create RAG service instance"""
    try:
        return RAGService()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG service is not available: {str(e)}")

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract and parse the JSON part from the text"""
    try:
        # First attempt: Check if the entire text is JSON
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    try:
        # Second attempt: Find ```json block
        json_block_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_block_match:
            return json.loads(json_block_match.group(1).strip())
    except json.JSONDecodeError:
        pass
    
    try:
        # Third attempt: Find JSON between curly braces
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    raise ValueError("Could not find a valid JSON in the response")

def validate_and_normalize_score(score: Any) -> float:
    """Validate and normalize the score to a range of 0-100"""
    try:
        score_float = float(score)
        return max(0.0, min(100.0, score_float))
    except (ValueError, TypeError):
        logger.warning(f"Invalid score format: {score}, using default value 50.0")
        return 50.0

def calculate_rag_weighted_score(base_score: float, rag_sources: List[Dict[str, Any]], confidence_factor: float = 1.0) -> float:
    """RAG 검색 결과를 바탕으로 점수에 가중치 적용"""
    if not rag_sources:
        return base_score * 0.7  # RAG 검색 결과가 없으면 신뢰도 감소
    
    # 유사도 점수들 추출
    similarities = []
    for source in rag_sources:
        if 'similarity' in source:
            try:
                sim_score = float(source['similarity'])
                similarities.append(sim_score)
            except (ValueError, TypeError):
                continue
    
    if similarities:
        # 평균 유사도를 가중치로 사용
        avg_similarity = statistics.mean(similarities)
        # 유사도가 높을수록 신뢰도 증가 (0.5-1.2 배율)
        weight_factor = 0.5 + (avg_similarity * 0.7)
        weighted_score = base_score * weight_factor * confidence_factor
        return max(0.0, min(100.0, weighted_score))
    
    return base_score * confidence_factor

def get_fallback_scores(text: str) -> Dict[str, float]:
    """RAG 실패시 기본 휴리스틱 점수 계산"""
    text_length = len(text.strip())
    word_count = len(text.split())
    
    # 기본 점수 계산
    grammar_score = 70.0  # 기본 문법 점수
    if text_length > 100:
        grammar_score += 10.0  # 충분한 길이
    if word_count > 10:
        grammar_score += 5.0   # 충분한 단어 수
    
    # 간단한 격식도 판단 (존댓말 검사)
    formality_score = 60.0
    if any(marker in text for marker in ['습니다', '됩니다', '있습니다']):
        formality_score += 20.0
    
    # 가독성 점수 (문장 길이 기반)
    sentences = text.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if avg_sentence_length < 20:  # 적당한 문장 길이
        readability_score = 75.0
    else:
        readability_score = max(40.0, 75.0 - (avg_sentence_length - 20) * 2)
    
    return {
        'grammarScore': grammar_score,
        'formalityScore': formality_score,
        'readabilityScore': readability_score
    }

@router.post("/analyze", response_model=QualityAnalysisResponse)
async def analyze_text_quality(
    request: QualityAnalysisRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)]
) -> QualityAnalysisResponse:
    """개선된 다중 평가 지표 기반 텍스트 품질 분석"""
    
    # 대상과 상황 정보 추출
    target_audience = request.target_audience
    context = request.context
    
    try:
        logger.info(f"품질 분석 시작 - 대상: {target_audience}, 맥락: {context}, 텍스트 길이: {len(request.text)}")
        
        # 대상 맞춤 분석 (핵심 기능)
        target_analysis = await _perform_target_specific_analysis(
            rag_service, request.text, target_audience, context
        )
        
        if target_analysis.get('success'):
            # RAG 기반 분석 성공
            scores = target_analysis['scores']
            final_scores = {
                'grammarScore': calculate_rag_weighted_score(
                    validate_and_normalize_score(scores.get('grammarScore', 70)),
                    target_analysis.get('sources', []), 1.0
                ),
                'formalityScore': calculate_rag_weighted_score(
                    validate_and_normalize_score(scores.get('formalityScore', 70)),
                    target_analysis.get('sources', []), 1.0
                ),
                'readabilityScore': calculate_rag_weighted_score(
                    validate_and_normalize_score(scores.get('readabilityScore', 70)),
                    target_analysis.get('sources', []), 1.0
                )
            }
            
            # 제안사항 추출
            suggestions = []
            for sugg in scores.get('suggestions', [])[:4]:
                suggestions.append(SuggestionItem(
                    original=sugg.get('original', ''),
                    suggestion=sugg.get('suggestion', ''),
                    reason=f"[{target_audience} 대상] {sugg.get('reason', '')}"
                ))
        else:
            # Fallback 점수 사용
            final_scores = get_fallback_scores(request.text)
            suggestions = [
                SuggestionItem(
                    original="전체 텍스트",
                    suggestion=f"{target_audience}에게 더 적합한 표현으로 수정해보세요",
                    reason="대상별 맞춤 분석을 위해 더 구체적인 피드백이 필요합니다"
                )
            ]
        
        logger.info(f"품질 분석 완료 - 최종 점수: {final_scores}")
        
        return QualityAnalysisResponse(
            grammarScore=final_scores['grammarScore'],
            formalityScore=final_scores['formalityScore'],
            readabilityScore=final_scores['readabilityScore'],
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"품질 분석 중 오류: {e}")
        
        # Fallback: 기본 휴리스틱 점수 사용
        fallback_scores = get_fallback_scores(request.text)
        fallback_suggestions = [
            SuggestionItem(
                original="전체 텍스트",
                suggestion="더 구체적인 분석을 위해 상황 맥락 정보를 제공해주세요",
                reason="RAG 기반 분석에 실패하여 기본 평가를 사용했습니다"
            )
        ]
        
        return QualityAnalysisResponse(
            grammarScore=fallback_scores['grammarScore'],
            formalityScore=fallback_scores['formalityScore'],
            readabilityScore=fallback_scores['readabilityScore'],
            suggestions=fallback_suggestions
        )

async def _perform_target_specific_analysis(rag_service, text: str, target_audience: str, context: str) -> Dict[str, Any]:
    """대상별 맞춤 품질 분석 - 단순화된 버전"""
    
    # 대상별 분석 프롬프트 구성
    audience_guide = {
        "초등학생": "초등학생이 이해하기 쉬운 쉬운 단어와 짧은 문장",
        "중학생": "중학생 수준의 어휘와 명확한 설명",
        "고등학생": "고등학생에게 적절한 수준의 어휘와 논리적 구조", 
        "대학생": "대학생 수준의 학술적이면서도 이해하기 쉬운 표현",
        "성인학습자": "성인 학습자를 위한 실용적이고 명확한 설명",
        "교사": "교사가 활용하기 좋은 교육적 표현",
        "학부모": "학부모가 이해하기 쉬운 친근하고 명확한 설명",
        "일반인": "일반인이 이해하기 쉬운 보편적 표현"
    }
    
    audience_description = audience_guide.get(target_audience, "일반적인 표현")
    context_note = f" ({context} 맥락에서)" if context != "일반" else ""
    
    prompt = f"""문해력 증진 및 교육 관련 문서를 참고하여, '{target_audience}' 대상{context_note}으로 다음 텍스트를 평가해주세요.

평가 기준:
- {audience_description}에 적합한가?
- 문법적으로 정확한가?
- 읽기 쉽고 이해하기 쉬운가?

반드시 다음 JSON 형식으로만 응답해주세요:
{{
    "grammarScore": <0-100 점수>,
    "formalityScore": <0-100 점수>,
    "readabilityScore": <0-100 점수>,
    "suggestions": [
        {{
            "original": "개선 필요한 부분",
            "suggestion": "개선된 표현",
            "reason": "개선 이유"
        }}
    ]
}}

분석할 텍스트: {text}"""
    
    try:
        result = await rag_service.ask_generative_question(query=prompt, context=f"{target_audience} 대상 분석")
        if result and result.get("success"):
            analysis_data = extract_json_from_text(result["answer"])
            return {
                'scores': analysis_data,
                'sources': result.get('sources', []),
                'success': True
            }
    except Exception as e:
        logger.warning(f"대상별 분석 실패: {e}")
    
    return {'scores': {}, 'sources': [], 'success': False}

async def _perform_learner_assessment(rag_service, text: str, target_audience: str, difficulty_level: str) -> Dict[str, Any]:
    """1단계: 학습자 맞춤 평가 - 대상 학습자의 이해 수준과 어휘 적합성"""
    prompt = f"""문해력 증진 및 교육 문서를 참고하여, 다음 텍스트가 '{target_audience}' 대상, '{difficulty_level}' 수준에 적합한지 평가해주세요.

평가 기준:
1. 어휘 난이도가 대상에게 적절한가?
2. 문장 구조가 이해하기 쉬운가?
3. 전문용어나 한자어 사용이 적절한가?

JSON 형식으로 응답:
{{
    "vocabulary_appropriateness": <0-100 점수>,
    "sentence_complexity": <0-100 점수>,
    "comprehension_level": <0-100 점수>,
    "suggestions": [
        {{
            "original": "어려운 표현",
            "suggestion": "쉬운 표현",
            "reason": "대상 수준에 맞지 않음"
        }}
    ]
}}

분석할 텍스트: {text}"""
    
    try:
        result = await rag_service.ask_generative_question(query=prompt, context="학습자 맞춤 평가")
        if result and result.get("success"):
            analysis_data = extract_json_from_text(result["answer"])
            return {
                'scores': analysis_data,
                'sources': result.get('sources', []),
                'success': True
            }
    except Exception as e:
        logger.warning(f"학습자 맞춤 평가 실패: {e}")
    
    return {'scores': {}, 'sources': [], 'success': False}

async def _perform_context_assessment(rag_service, text: str, context: str, purpose: str) -> Dict[str, Any]:
    """2단계: 맥락 맞춤 평가 - 스타일별 논리 전개와 전달력"""
    
    context_prompts = {
        '교육': "교육용 텍스트 작성 가이드",
        '보고서': "보고서 작성 가이드", 
        '공문': "공문 작성 가이드",
        '일반': "일반 커뮤니케이션 가이드"
    }
    
    doc_reference = context_prompts.get(context, "일반 글쓰기 가이드")
    
    prompt = f"""{doc_reference} 문서를 참고하여, '{context}' 맥락에서 '{purpose}' 목적의 다음 텍스트를 평가해주세요.

평가 기준:
1. 논리적 구조와 전개가 적절한가?
2. 목적에 맞는 내용 전달이 되고 있는가?
3. 해당 맥락의 관례와 형식을 따르고 있는가?

JSON 형식으로 응답:
{{
    "logical_structure": <0-100 점수>,
    "purpose_alignment": <0-100 점수>, 
    "format_compliance": <0-100 점수>,
    "suggestions": [
        {{
            "original": "개선 필요 부분",
            "suggestion": "맥락에 맞는 표현",
            "reason": "해당 맥락의 관례에 맞지 않음"
        }}
    ]
}}

분석할 텍스트: {text}"""

    try:
        result = await rag_service.ask_generative_question(query=prompt, context="맥락 맞춤 평가")
        if result and result.get("success"):
            analysis_data = extract_json_from_text(result["answer"])
            return {
                'scores': analysis_data,
                'sources': result.get('sources', []),
                'success': True
            }
    except Exception as e:
        logger.warning(f"맥락 맞춤 평가 실패: {e}")
    
    return {'scores': {}, 'sources': [], 'success': False}

async def _perform_style_assessment(rag_service, text: str, context: str) -> Dict[str, Any]:
    """3단계: 스타일 맞춤 평가 - 자연스러움과 네거티브 프롬프트 준수"""
    prompt = f"""문법 규칙 및 스타일 가이드 문서를 참고하여, 다음 텍스트의 자연스러움과 부적절한 표현 여부를 평가해주세요.

평가 기준:
1. 문법적 정확성
2. 자연스러운 표현인가?
3. 부적절하거나 피해야 할 표현이 있는가?
4. 일관성 있는 문체인가?

JSON 형식으로 응답:
{{
    "grammar_accuracy": <0-100 점수>,
    "naturalness": <0-100 점수>,
    "appropriateness": <0-100 점수>,
    "consistency": <0-100 점수>,
    "suggestions": [
        {{
            "original": "부자연스러운 표현",
            "suggestion": "자연스러운 표현",
            "reason": "문법 오류 또는 부자연스러움"
        }}
    ]
}}

분석할 텍스트: {text}"""

    try:
        result = await rag_service.ask_generative_question(query=prompt, context="스타일 맞춤 평가")
        if result and result.get("success"):
            analysis_data = extract_json_from_text(result["answer"])
            return {
                'scores': analysis_data,
                'sources': result.get('sources', []),
                'success': True
            }
    except Exception as e:
        logger.warning(f"스타일 맞춤 평가 실패: {e}")
    
    return {'scores': {}, 'sources': [], 'success': False}

def _integrate_assessment_scores(learner_assessment: Dict, context_assessment: Dict, style_assessment: Dict) -> Dict[str, float]:
    """다중 평가 결과를 통합하여 최종 점수 계산"""
    
    # 가중치 설정 (교육 도메인 특성상 학습자 맞춤을 우선시)
    learner_weight = 0.4
    context_weight = 0.3  
    style_weight = 0.3
    
    # 학습자 맞춤 평가 점수 추출
    learner_scores = learner_assessment.get('scores', {})
    vocab_score = validate_and_normalize_score(learner_scores.get('vocabulary_appropriateness', 70))
    complexity_score = validate_and_normalize_score(learner_scores.get('sentence_complexity', 70))
    comprehension_score = validate_and_normalize_score(learner_scores.get('comprehension_level', 70))
    
    # 맥락 맞춤 평가 점수 추출  
    context_scores = context_assessment.get('scores', {})
    logic_score = validate_and_normalize_score(context_scores.get('logical_structure', 70))
    purpose_score = validate_and_normalize_score(context_scores.get('purpose_alignment', 70))
    format_score = validate_and_normalize_score(context_scores.get('format_compliance', 70))
    
    # 스타일 맞춤 평가 점수 추출
    style_scores = style_assessment.get('scores', {})
    grammar_score = validate_and_normalize_score(style_scores.get('grammar_accuracy', 70))
    natural_score = validate_and_normalize_score(style_scores.get('naturalness', 70))
    appropriate_score = validate_and_normalize_score(style_scores.get('appropriateness', 70))
    consistent_score = validate_and_normalize_score(style_scores.get('consistency', 70))
    
    # 통합 점수 계산
    integrated_grammar = (
        comprehension_score * learner_weight +
        logic_score * context_weight +
        grammar_score * style_weight
    )
    
    integrated_formality = (
        vocab_score * learner_weight +
        format_score * context_weight +
        appropriate_score * style_weight  
    )
    
    integrated_readability = (
        complexity_score * learner_weight +
        purpose_score * context_weight +
        natural_score * style_weight
    )
    
    return {
        'grammar': integrated_grammar,
        'formality': integrated_formality,
        'readability': integrated_readability
    }

def _generate_integrated_suggestions(learner_assessment: Dict, context_assessment: Dict, style_assessment: Dict, original_text: str) -> List[SuggestionItem]:
    """세 가지 평가 결과를 종합하여 통합 개선 제안 생성"""
    suggestions = []
    
    # 각 평가에서 제안사항 수집
    for assessment, label in [
        (learner_assessment, "학습자 맞춤"),
        (context_assessment, "맥락 맞춤"), 
        (style_assessment, "스타일 맞춤")
    ]:
        if assessment.get('success') and assessment.get('scores', {}).get('suggestions'):
            for sugg in assessment['scores']['suggestions'][:2]:  # 각 평가에서 최대 2개씩
                suggestions.append(SuggestionItem(
                    original=sugg.get('original', ''),
                    suggestion=sugg.get('suggestion', ''),
                    reason=f"[{label}] {sugg.get('reason', '')}"
                ))
    
    # 제안이 없을 경우 기본 제안 추가
    if not suggestions:
        suggestions.append(SuggestionItem(
            original="전체 텍스트",
            suggestion="문장을 더 명확하고 이해하기 쉽게 작성해보세요",
            reason="구체적인 개선점을 찾지 못했지만, 전반적인 명확성 향상을 권장합니다"
        ))
    
    # 최대 4개 제안으로 제한
    return suggestions[:4]


@router.post("/suggestions", response_model=ContextSuggestionsResponse)
async def get_context_suggestions(
    request: ContextSuggestionsRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)]
) -> ContextSuggestionsResponse:
    """맥락별 표현 개선 제안 (RAG 기반)"""
    try:
        # RAG에 보낼 상세한 프롬프트(질문) 구성
        prompt = f"""'글쓰기 스타일 가이드' 문서를 참고하여, '{request.context}' 맥락에서 아래 텍스트의 표현을 더 좋게 바꿀 수 있는 제안들을 3개만 찾아줘.
        
        반드시 아래의 JSON 형식으로만 응답해야 해. 다른 설명은 절대 추가하지 마.
        [
          {{
            "original": "개선이 필요한 원본 단어 또는 구절",
            "suggestion": "더 나은 표현 제안",
            "reason": "왜 그렇게 제안하는지에 대한 간단한 이유"
          }}
        ]

        --- 개선할 텍스트 ---
        {request.text}
        """
        
        # RAG 서비스 호출 (생성형 답변 함수 사용)
        rag_result = await rag_service.ask_generative_question(query=prompt, context="표현 개선 제안")

        if not rag_result or not rag_result.get("success"):
            raise HTTPException(status_code=500, detail="RAG 서비스가 표현 제안에 실패했습니다.")

        # RAG의 답변(answer)에 포함된 JSON 문자열을 파싱
        suggestions_list = json.loads(rag_result["answer"])
        
        relevant_suggestions = [SuggestionItem(**s) for s in suggestions_list]

        return ContextSuggestionsResponse(
            suggestions=relevant_suggestions,
            count=len(relevant_suggestions)
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="RAG 서비스의 응답(JSON)을 파싱하는 데 실패했습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"맥락 제안 생성 실패: {str(e)}")