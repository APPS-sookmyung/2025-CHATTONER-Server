"""
OpenAI API 서비스
텍스트 스타일 변환을 위한 OpenAI GPT 모델 호출
"""

import os
from typing import Dict, List, Any
from openai import OpenAI
import json

class OpenAIService:
    """OpenAI API 호출 관리 클래스"""
    
    def __init__(self, api_key=None, model=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        # GPT-4o 최신 모델 사용
        self.model = model or "gpt-4o"
    #3가지 스타일(direct, gentle, neutral)을 변환하는 메인 함수.
    def convert_text_styles(self, input_text: str, prompts: Dict[str, str]) -> Dict[str, str]:
        """
        입력 텍스트를 3가지 스타일로 변환
        
        Args:
            input_text: 변환할 원본 텍스트
            prompts: 각 스타일별 프롬프트 딕셔너리
        
        Returns:
            3가지 스타일 변환 결과 딕셔너리
        """
        results = {}
        
        try:
            # Direct 스타일 변환
            if 'direct' in prompts:
                results['direct'] = self._convert_single_style(
                    input_text, prompts['direct']
                )
            
            # Gentle 스타일 변환
            if 'gentle' in prompts:
                results['gentle'] = self._convert_single_style(
                    input_text, prompts['gentle']
                )
            
            # Neutral 스타일 변환
            if 'neutral' in prompts:
                results['neutral'] = self._convert_single_style(
                    input_text, prompts['neutral']
                )
            
            return results
            
        except Exception as e:
            print(f"OpenAI API 호출 오류: {e}")
            # 오류 시 원본 텍스트 반환
            return {
                'direct': input_text,
                'gentle': input_text,
                'neutral': input_text
            }
    
    # 단일 스타일 변환을 수행하는 헬퍼 함수
    # 바로 위 함수는 내부적으로 _convert_single_style() 함수를 호출하여 각각 변환.
    def _convert_single_style(self, input_text: str, prompt: str) -> str:
        """
        단일 스타일 변환 수행
        
        Args:
            input_text: 변환할 텍스트
            prompt: 변환에 사용할 프롬프트
        
        Returns:
            변환된 텍스트
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": f"다음 텍스트를 변환해주세요:\n\n{input_text}"
                    }
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            converted_text = response.choices[0].message.content.strip()
            return converted_text
            
        except Exception as e:
            print(f"단일 스타일 변환 오류: {e}")
            return input_text
    
    #사용자 텍스트의 감정을 수치화 (15점, 신뢰도 01).
    #결과는 JSON 형태로 요청하고, 이를 파싱하여 반환.
    #예외 시 기본값 3점 / 0.5 신뢰도 반환.


    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        텍스트 감정 분석
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            감정 분석 결과 (rating, confidence)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 감정 분석 전문가입니다. 
                        텍스트의 감정을 분석하여 1-5점 평점과 0-1 신뢰도를 제공하세요.
                        응답은 JSON 형식으로: {"rating": number, "confidence": number}"""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            parsed_result = json.loads(result)
            
            return {
                "rating": max(1, min(5, round(parsed_result.get("rating", 3)))),
                "confidence": max(0, min(1, parsed_result.get("confidence", 0.5)))
            }
            
        except Exception as e:
            print(f"감정 분석 오류: {e}")
            return {"rating": 3, "confidence": 0.5}
    
    async def analyze_style_feedback(self, feedback_text: str) -> Dict[str, float]:
        """
        사용자 피드백을 분석하여 스타일 조정값 계산
        
        Args:
            feedback_text: 사용자 피드백 텍스트
        
        Returns:
            스타일 조정값 딕셔너리
        """
        try: # 격식, 친근함, 감정, 직설성을 조정함. 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": """사용자 피드백을 분석하여 스타일 조정값을 계산하세요.
                        다음 JSON 형식으로 응답: 
                        {
                            "formalityDelta": float (-1.0 to 1.0),
                            "friendlinessDelta": float (-1.0 to 1.0), 
                            "emotionDelta": float (-1.0 to 1.0),
                            "directnessDelta": float (-1.0 to 1.0)
                        }
                        
                        피드백 분석 기준:
                        - "더 격식있게", "공식적으로" → formalityDelta 양수
                        - "더 친근하게", "부드럽게" → friendlinessDelta 양수  
                        - "더 감정적으로", "따뜻하게" → emotionDelta 양수
                        - "더 직설적으로", "명확하게" → directnessDelta 양수
                        """
                    },
                    {
                        "role": "user",
                        "content": feedback_text # 피드백 텍스트를 입력
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            print(f"피드백 분석 오류: {e}")
            return {
                "formalityDelta": 0.0,
                "friendlinessDelta": 0.0, 
                "emotionDelta": 0.0,
                "directnessDelta": 0.0
            }