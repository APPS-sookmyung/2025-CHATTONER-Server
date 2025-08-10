"""
LoRA 파인튜닝 모델을 활용한 공식 문서 변환 체인 - Gemma-2 적용 버전
공적인 말투 변환 시 LoRA 모델 1차 변환 후 ChatGPT 2차 다듬기 적용

주요 기능:
- Gemma-2-2b-it 모델 + 4bit 양자화
- 사용자 프로필과 컨텍스트 기반 공식 문서 변환 조건 판단
- LoRA 모델을 통한 1차 공식 말투 변환
- ChatGPT를 통한 2차 다듬기 및 보완
- Services와 동일한 구조로 프롬프트 엔지니어링 통합
"""

import sys
import logging
from pathlib import Path
import os
from typing import Dict, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 프로젝트 경로 설정
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 로거 설정
logger = logging.getLogger(__name__)

class FinetuneChain:
    """LoRA 파인튜닝 모델을 활용한 공식 문서 변환 체인"""
    
    def __init__(
    self,
    model_name: str = "gpt-4o",
    temperature: float = 0.3,
    prompt_engineer: Optional[object] = None,
    openai_service: Optional[object] = None,
):
        """Finetune Chain 초기화"""
        dotenv_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(dotenv_path=dotenv_path)
        
       # Services 초기화 (DI 우선, 없으면 내부 생성)
        try:
            if prompt_engineer is None or openai_service is None:
                from services.prompt_engineering import PromptEngineer
                from services.openai_services import OpenAIService
                self.prompt_engineer = prompt_engineer or PromptEngineer()
                self.openai_service = openai_service or OpenAIService()
            else:
                self.prompt_engineer = prompt_engineer
                self.openai_service = openai_service
            self.services_available = True
            logger.info("Services 초기화 완료")
        except ImportError as e:
            self.services_available = False
            logger.warning(f"Services import 실패: {e}")
        except Exception as e:
            self.services_available = False
            logger.error(f"Services 인스턴스 생성 실패: {e}")
        """
        # OpenAI LLM 설정
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )"""
        
        # LoRA 모델 관련 설정
        self.lora_model_path = Path(__file__).resolve().parents[1] / "finetuned_models"
        self.base_model = None
        self.lora_model = None
        self.tokenizer = None
        self.is_lora_loaded = False
        
        self._load_lora_model()
    
    def _load_lora_model(self):
        """LoRA 모델 로드 - Gemma-2 적용"""
        try:
            # Hugging Face 토큰 확인
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                logger.warning("HUGGINGFACE_TOKEN이 설정되지 않음")
            
            # Gemma-2 모델 사용
            base_model_name = "google/gemma-2-2b-it"
            
            # 4bit 양자화 설정
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            logger.info(f"Gemma-2 모델 로드 중: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                token=hf_token
            )
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                token=hf_token,
                torch_dtype=torch.float16
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # LoRA 어댑터 로드
            adapter_loaded = False
            if self.lora_model_path.exists():
                adapter_config_path = self.lora_model_path / "adapter_config.json"
                if adapter_config_path.exists():
                    logger.info(f"LoRA 어댑터 로드 중: {self.lora_model_path}")
                    self.lora_model = PeftModel.from_pretrained(
                        self.base_model, str(self.lora_model_path)
                    )
                    adapter_loaded = True
                else:
                    logger.warning("adapter_config.json이 없어 기본 모델만 사용")
                    self.lora_model = self.base_model
            else:
                logger.warning("LoRA 어댑터 경로가 없어 기본 모델만 사용")
                self.lora_model = self.base_model

            self.is_lora_loaded = adapter_loaded
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self.is_lora_loaded = False
    
    def _should_use_lora(self, user_profile: Dict, context: str) -> bool:
        """공식 문서 변환 필요 여부 판단 - 수정된 버전"""
        # 1. 직접적인 공식 문서 모드
        if user_profile.get('formal_document_mode', False):
            return True
        
        # 2. 격식도 기반 판단
        formality_level = user_profile.get('sessionFormalityLevel', 
                                          user_profile.get('baseFormalityLevel', 3))
        
        # 격식도가 5 이상
        if formality_level >= 5:
            return True
        
        # 격식도 4 이상 & business/report 컨텍스트
        if formality_level >= 4 and context in ["business", "report"]:
            return True
        
    
    def _generate_with_lora(self, input_text: str, max_length: int = 512) -> str:
        """LoRA 모델로 1차 변환 - Gemma-2 템플릿"""
        if not self.is_lora_loaded:
            raise Exception("LoRA 모델이 로드되지 않았습니다.")
        
        try:
            # Gemma-2 프롬프트 템플릿
            prompt_template = f"""<start_of_turn>user
다음 텍스트를 공식적이고 정중한 말투로 변환해주세요:

{input_text}<end_of_turn>
<start_of_turn>model
"""
            
            # 토크나이징
            inputs = self.tokenizer(
                prompt_template,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # GPU로 이동
            if torch.cuda.is_available():
                try:
                    target_device = getattr(self.lora_model, "device", None)  # 안전하게 가져오기
                    if target_device is not None:
                            inputs = {k: v.to(target_device) for k, v in inputs.items()}
                except Exception:
                    # device_map="auto"가 알아서 장치 배치를 처리하게 두기
                        pass

            
            # 생성 (Gemma-2 최적화 파라미터)
            with torch.no_grad():
                outputs = self.lora_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Gemma-2 응답 추출
            if "<start_of_turn>model" in generated_text:
                result = generated_text.split("<start_of_turn>model")[-1].strip()
            else:
                result = generated_text[len(prompt_template):].strip()
            
            return result
            
        except Exception as e:
            logger.error(f"LoRA 생성 실패: {e}")
            return input_text # 실패 시 원본 텍스트 반환
    
    async def _refine_with_gpt(self, lora_output: str, user_profile: Dict, context: str) -> str:
        """ChatGPT로 2차 다듬기 (기존 OpenAIService 패턴 사용)"""
        if not self.services_available:
            return lora_output
        
        try:
            # 프롬프트엔지니어링_프롬프트 생성
            negative_preferences = user_profile.get('negativePreferences', {})
            prompts = self.prompt_engineer.generate_conversion_prompts(
                user_profile=user_profile,
                context=context,
                negative_preferences=negative_preferences
            )
            
            refinement_prompt = f"""
다음은 LoRA 모델이 생성한 공식 문서 변환 결과입니다.
이를 더욱 자연스럽고 완성도 높은 공식 문서로 다듬어주세요.

{prompts.get('neutral', '')}

LoRA 변환 결과:
{lora_output}
"""
            
            refined_result = self.openai_service._convert_single_style(
                input_text=lora_output,
                prompt=refinement_prompt
            )
            
            return refined_result
            
        except Exception as e:
            logger.error(f"GPT 다듬기 실패: {e}")
            return lora_output
    
    async def convert_to_formal(self, 
                               input_text: str, 
                               user_profile: Dict, 
                               context: str = "business",
                               force_convert: bool = False) -> Dict:
        """
        공식 문서 변환 (LoRA + ChatGPT 파이프라인)
        
        Args:
            input_text: 변환할 텍스트
            user_profile: 사용자 프로필
            context: 컨텍스트 ("business", "report", "personal" 등)
            force_convert: 사용자가 명시적으로 "공식문서로 변환" 클릭한 경우 True
        """
        
        # 우선순위 1: 사용자 명시적 요청
        if force_convert:
            logger.info("사용자 명시적 요청으로 공식 문서 변환 강제 실행")
            should_convert = True
            conversion_reason = "user_explicit_request"
        else:
            # 우선순위 2: 자동 조건 판단
            should_convert = self._should_use_lora(user_profile, context)
            conversion_reason = "auto_condition" if should_convert else "condition_not_met"
        
        # 변환 조건에 해당하지 않는 경우
        if not should_convert:
            return {
                "success": False,
                "error": "공식 문서 변환 조건에 해당하지 않습니다.",
                "converted_text": "",
                "method": "none",
                "reason": conversion_reason
            }
        
        try:
            # 1차: LoRA 모델 변환
            if self.is_lora_loaded:
                logger.info("LoRA 모델을 통한 1차 변환 시작")
                lora_output = self._generate_with_lora(input_text)
                method = "lora_gpt"
            else:
                logger.warning("LoRA 모델 미사용, ChatGPT만 사용")
                lora_output = input_text
                method = "gpt_only"
            
            # 2차: ChatGPT 다듬기
            logger.info("ChatGPT를 통한 2차 다듬기 시작")
            final_output = await self._refine_with_gpt(lora_output, user_profile, context)
            
            return {
                "success": True,
                "converted_text": final_output,
                "lora_output": lora_output if self.is_lora_loaded else None,
                "method": method,
                "reason": conversion_reason,
                "forced": force_convert,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"공식 문서 변환 실패: {e}")
            return {
                "success": False,
                "error": f"변환 중 오류 발생: {str(e)}",
                "converted_text": "",
                "method": "error",
                "reason": conversion_reason
            }
    
    async def convert_by_user_request(self, 
                                     input_text: str, 
                                     user_profile: Dict, 
                                     context: str = "business") -> Dict:
        """사용자가 '공식문서로 변환' 버튼을 클릭했을 때 사용하는 메서드"""
        return await self.convert_to_formal(
            input_text=input_text,
            user_profile=user_profile,
            context=context,
            force_convert=True  # 항상 강제 변환
        )
    
    def get_status(self) -> Dict:
        """상태 정보"""
        return {
            "lora_status": "ready" if self.is_lora_loaded else "not_ready",
            "lora_model_path": str(self.lora_model_path),
            "services_available": self.services_available,
            "base_model_loaded": self.base_model is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_name": "gemma-2-2b-it"
        }