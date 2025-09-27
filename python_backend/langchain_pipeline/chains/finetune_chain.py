"""
LoRA Fine-tuning Model-based Formal Document Conversion Chain - HTTP Client Version
Applies RunPod inference server primary conversion followed by ChatGPT secondary refinement for formal tone conversion

Key Features:
- RunPod inference server HTTP requests
- User profile and context-based formal document conversion condition determination
- Primary formal tone conversion through RunPod server
- Secondary refinement and supplementation through ChatGPT
- Prompt engineering integration with same structure as Services
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import httpx

# Project path configuration
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Logger configuration
logger = logging.getLogger(__name__)

class FinetuneChain:
    """Formal document conversion chain utilizing LoRA fine-tuning model"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        prompt_engineer: Optional[object] = None,
        openai_service: Optional[object] = None,
    ):
        """Initialize Finetune Chain"""
        dotenv_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(dotenv_path=dotenv_path)
        
        # RunPod inference server URL configuration
        from core.config import get_settings
        settings = get_settings()
        self.inference_url = settings.FINETUNE_URL
        
        # Services initialization (DI first, internal creation if not available)
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
            logger.info("Services initialization completed")
        except ImportError as e:
            self.services_available = False
            logger.warning(f"Services import failed: {e}")
        except Exception as e:
            self.services_available = False
            logger.error(f"Services instance creation failed: {e}")
     
        # Check RunPod server connection status
        self.is_inference_server_available = self._check_inference_server()
    
    def _check_inference_server(self) -> bool:
        """Check RunPod inference server connection status"""
        try:
            response = httpx.get(f"{self.inference_url}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info("RunPod inference server connection successful")
                return True
            else:
                logger.warning(f"RunPod inference server response error: {response.status_code}")
                return False
        except httpx.HTTPError as e:
            logger.warning(f"RunPod inference server connection failed: {e}")
            return False
    
    def _should_use_lora(self, user_profile: Dict, context: str) -> bool:
        """Determine if formal document conversion is needed"""
        # 1. Direct formal document mode
        if user_profile.get('formal_document_mode', False):
            return True
        
        # 2. Formality level-based determination
        formality_level = user_profile.get('sessionFormalityLevel', 
                                          user_profile.get('baseFormalityLevel', 3))
        
        # Formality level 5 or higher
        if formality_level >= 5:
            return True
        
        # Formality level 4 or higher & business/report context
        if formality_level >= 4 and context in ["business", "report"]:
            return True
        
        return False
    
    async def _generate_with_lora(self, input_text: str, max_tokens: int = 256) -> str:
        """Primary conversion through RunPod inference server"""
        if not self.is_inference_server_available:
            raise Exception("Cannot connect to RunPod inference server.")
        
        try:
            # HTTP request to RunPod inference server
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.inference_url}/generate",
                    json={
                        "prompt": input_text,
                        "max_new_tokens": max_tokens,
                        "temperature": 0.7,
                        "do_sample": True,
                    },
                )
                response.raise_for_status()
                result = response.json()
            
            return result["result"]
            
        except httpx.HTTPError as e:
            logger.error(f"RunPod server request failed: {e}")
            return input_text  # Return original text on failure
        except Exception as e:
            logger.error(f"RunPod server response processing failed: {e}")
            return input_text  # Return original text on failure
    
    async def _refine_with_gpt(self, lora_output: str, original_text:str, user_profile: Dict, context: str) -> str:
        """Secondary refinement with ChatGPT (using existing OpenAIService pattern)"""
        if not self.services_available:
            return lora_output
        
        try:
            # Generate prompt engineering prompts
            negative_preferences = user_profile.get('negativePreferences', {})
            prompts = self.prompt_engineer.generate_conversion_prompts(
                user_profile=user_profile,
                context=context,
                negative_preferences=negative_preferences
            )
            
            # Determine style based on user profile
            directness_level = user_profile.get('sessionDirectnessLevel', 
                                              user_profile.get('baseDirectnessLevel', 3))
            
            # Style selection based on directness level 
            if directness_level >= 4:
                style_key = 'direct'   
            elif directness_level <= 2:
                style_key = 'gentle'  
            else:
                style_key = 'neutral'

            refinement_prompt = f"""
This is a formal document conversion task.

【Original Text】
{original_text}

【Primary LoRA Conversion Result】  
{lora_output}

【Task Instructions】
{prompts.get(style_key, prompts.get('neutral', ''))}

Please refine the primary conversion result into a more natural and polished formal document while maintaining the original intent, referencing both the original text and primary conversion result.

- Preserve the core meaning and context of the original text
- Maintain the formal tone of the primary conversion but improve unnatural parts
- Supplement any missing information by referencing the original
"""

            
            refined_result = self.openai_service._convert_single_style(
                input_text=lora_output,
                prompt=refinement_prompt
            )
            
            return refined_result
            
        except Exception as e:
            logger.error(f"GPT refinement failed: {e}")
            return lora_output
    
    async def convert_to_formal(self, 
                               input_text: str, 
                               user_profile: Dict, 
                               context: str = "business",
                               force_convert: bool = False) -> Dict:
        """
        Formal document conversion (RunPod inference server + ChatGPT pipeline)
        
        Args:
            input_text: Text to convert
            user_profile: User profile
            context: Context ("business", "report", "personal", etc.)
            force_convert: True when user explicitly clicks "Convert to Formal Document"
        """
        
        # Priority 1: User explicit request
        if force_convert:
            logger.info("Forcing formal document conversion due to user explicit request")
            should_convert = True
            conversion_reason = "user_explicit_request"
        else:
            # Priority 2: Automatic condition determination
            should_convert = self._should_use_lora(user_profile, context)
            conversion_reason = "auto_condition" if should_convert else "condition_not_met"
        
        # When conversion conditions are not met
        if not should_convert:
            return {
                "success": False,
                "error": "Does not meet formal document conversion conditions.",
                "converted_text": "",
                "method": "none",
                "reason": conversion_reason
            }
        
        try:
            # Primary: RunPod inference server conversion
            if self.is_inference_server_available:
                logger.info("Starting primary conversion through RunPod inference server")
                lora_output = await self._generate_with_lora(input_text)
                method = "lora_gpt"
            else:
                logger.warning("RunPod inference server not used, using ChatGPT only")
                lora_output = input_text
                method = "gpt_only"
            
            # Secondary: ChatGPT refinement
            logger.info("Starting secondary refinement through ChatGPT")
            final_output = await self._refine_with_gpt(
                lora_output = lora_output,
                original_text=input_text,
                user_profile=user_profile,
                context=context)
            
            return {
                "success": True,
                "converted_text": final_output,
                "lora_output": lora_output if self.is_inference_server_available else None,
                "method": method,
                "reason": conversion_reason,
                "forced": force_convert,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Formal document conversion failed: {e}")
            return {
                "success": False,
                "error": f"Error occurred during conversion: {str(e)}",
                "converted_text": "",
                "method": "error",
                "reason": conversion_reason
            }
    
    async def convert_by_user_request(self, 
                                     input_text: str, 
                                     user_profile: Dict, 
                                     context: str = "business") -> Dict:
        """Method used when user clicks 'Convert to Formal Document' button"""
        return await self.convert_to_formal(
            input_text=input_text,
            user_profile=user_profile,
            context=context,
            force_convert=True  # Always force conversion
        )
    
    async def convert_to_business(
        self,
        input_text: str,
        user_profile: Dict
    ) -> Dict:
        """Convert to business style (for button click)"""
        return await self.convert_by_user_request(input_text, user_profile, "business")

    async def convert_to_report(
        self,
        input_text: str,
        user_profile: Dict
    ) -> Dict:
        """Convert to report style (for button click)"""
        return await self.convert_by_user_request(input_text, user_profile, "report")
    
    def get_status(self) -> Dict:
        """Status information"""
        return {
            "lora_status": "ready" if self.is_inference_server_available else "not_ready",
            "lora_model_path": "runpod_server",
            "services_available": self.services_available,
            "base_model_loaded": self.is_inference_server_available,
            "device": "runpod_gpu",
            "model_name": "gemma-2-2b-it",
            "inference_url": self.inference_url
        }