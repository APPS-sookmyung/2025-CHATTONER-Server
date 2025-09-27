"""
FinetuneChain Test Script - Version with force_convert flag
Test LoRA fine-tuning model and ChatGPT 2-stage conversion pipeline

Execution method:
python test_finetune_chain.py
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Project path configuration
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from langchain_pipeline.chains.finetune_chain import FinetuneChain

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_should_use_lora():
    """Test LoRA usage conditions (modified logic)"""
    print("\nLoRA Usage Condition Unit Test")
    print("=" * 40)
    
    try:
        chain = FinetuneChain.__new__(FinetuneChain)
        
        test_cases = [
            # Formality level 5 or higher - always True
            ({"baseFormalityLevel": 5}, "personal", True),
            ({"baseFormalityLevel": 5}, "business", True),
            
            # Formality level 4 + business/report - True
            ({"baseFormalityLevel": 4}, "business", True),
            ({"baseFormalityLevel": 4}, "report", True),
            ({"baseFormalityLevel": 4}, "personal", False),  # personal is not allowed
            
            # Formality level 3 - False even for business/report
            ({"baseFormalityLevel": 3}, "business", False),
            ({"baseFormalityLevel": 3}, "report", False),
            ({"baseFormalityLevel": 3}, "personal", False),
            
            # formal_document_mode = True - always True
            ({"baseFormalityLevel": 2, "formal_document_mode": True}, "personal", True),
            ({"baseFormalityLevel": 1, "formal_document_mode": True}, "business", True),
            
            # sessionFormalityLevel takes priority
            ({"sessionFormalityLevel": 5, "baseFormalityLevel": 2}, "business", True),
            ({"sessionFormalityLevel": 3, "baseFormalityLevel": 5}, "business", False),
            
            # Empty profile (default 3) - False
            ({}, "business", False),
            ({}, "report", False),
            ({}, "personal", False),
        ]
        
        for profile, context, expected in test_cases:
            result = chain._should_use_lora(profile, context)
            status = "PASS" if result == expected else "FAIL"
            print(f"[{status}] 프로필: {profile}, 컨텍스트: {context} -> {result} (예상: {expected})")
            
    except Exception as e:
        print(f"[ERROR] Unit test failed: {e}")

async def test_force_convert():
    """Test user explicit request"""
    print("\nUser Explicit Request Test")
    print("=" * 60)
    
    try:
        finetune_chain = FinetuneChain()
    except Exception as e:
        print(f"[ERROR] FinetuneChain initialization failed: {e}")
        return
    
    # Cases that originally don't meet conversion conditions
    force_test_cases = [
        {
            "input": "안녕! 오늘 뭐해?",
            "profile": {"baseFormalityLevel": 1},  
            "context": "personal",  
            "description": "Casual + personal conversation (originally no conversion)"
        },
        {
            "input": "ㅋㅋㅋ 재밌네 ㅎㅎ 😂",
            "profile": {"baseFormalityLevel": 2},
            "context": "casual",
            "description": "Casual message with emoticons"
        },
        {
            "input": "빨리빨리 해주세요!!",
            "profile": {},  
            "context": "personal",
            "description": "Empty profile + urgent request"
        }
    ]
    
    for i, test_case in enumerate(force_test_cases, 1):
        print(f"\n[Force Conversion Test {i}] {test_case['description']}")
        print(f"원본: {test_case['input']}")
        
        # 1. Normal conversion (force_convert=False) - expected to fail
        try:
            result_normal = await finetune_chain.convert_to_formal(
                input_text=test_case['input'],
                user_profile=test_case['profile'],
                context=test_case['context'],
                force_convert=False
            )
            
            if result_normal['success']:
                print("Normal conversion: [SUCCESS] (unexpected)")
                print(f"   Result: {result_normal['converted_text']}")
            else:
                print(f"Normal conversion: [FAIL] {result_normal['error']}")
        except Exception as e:
            print(f"Normal conversion: [ERROR] {e}")
        
        # 2. Force conversion (force_convert=True) - expected to succeed
        try:
            result_forced = await finetune_chain.convert_to_formal(
                input_text=test_case['input'],
                user_profile=test_case['profile'],
                context=test_case['context'],
                force_convert=True
            )
            
            if result_forced['success']:
                print(f"Force conversion: [SUCCESS] ({result_forced['method']})")
                print(f"   Result: {result_forced['converted_text']}")
                print(f"   Conversion reason: {result_forced['reason']}")
                print(f"   Force mode: {result_forced['forced']}")
            else:
                print(f"Force conversion: [FAIL] {result_forced['error']}")
        except Exception as e:
            print(f"Force conversion: [ERROR] {e}")
        
        print("-" * 60)

async def test_convenience_method():
    """Test convenience methods"""
    print("\nConvenience Method Test")
    print("=" * 40)
    
    try:
        finetune_chain = FinetuneChain()
    except Exception as e:
        print(f"[ERROR] FinetuneChain initialization failed: {e}")
        return
    
    test_input = "야 이거 언제 하냐?"
    test_profile = {"baseFormalityLevel": 1}
    
    try:
        # Test convert_by_user_request method
        result = await finetune_chain.convert_by_user_request(
            input_text=test_input,
            user_profile=test_profile,
            context="personal"
        )
        
        if result['success']:
            print("[SUCCESS] Convenience method")
            print(f"Original: {test_input}")
            print(f"Converted: {result['converted_text']}")
            print(f"Force conversion: {result['forced']}")
            print(f"Conversion reason: {result['reason']}")
            print(f"Method used: {result['method']}")
        else:
            print(f"[FAIL] Convenience method: {result['error']}")
    except Exception as e:
        print(f"[ERROR] Convenience method: {e}")

async def test_finetune_chain():
    """FinetuneChain integration test"""
    print("=" * 60)
    print("FinetuneChain Integration Test Started")
    print("=" * 60)
    
    # 1. Initialize FinetuneChain
    try:
        print("\nInitializing FinetuneChain...")
        finetune_chain = FinetuneChain()
        print("[SUCCESS] FinetuneChain initialization completed")
    except Exception as e:
        print(f"[ERROR] FinetuneChain initialization failed: {e}")
        return
    
    # 2. Check system status
    print("\nSystem Status Check")
    print("=" * 40)
    status = finetune_chain.get_status()
    
    print(f"LoRA status: {status['lora_status']}")
    print(f"Services available: {status['services_available']}")
    print(f"Base model loaded: {status['base_model_loaded']}")
    print(f"Device: {status['device']}")
    print(f"LoRA model path: {status['lora_model_path']}")
    print(f"Model name: {status.get('model_name', 'N/A')}")
    
    # 3. Set test user profiles
    test_user_profiles = {
        "high_formal": {
            "baseFormalityLevel": 5, 
            "baseFriendlinessLevel": 3,
            "baseEmotionLevel": 2,    
            "baseDirectnessLevel": 4,
            "negativePreferences": {
                "avoidFloweryLanguage": "moderate",
                "avoidRepetitiveWords": "strict",
                "emoticonUsage": "strict"
            },
            "formal_document_mode": True
        },
        "medium_formal": {
            "baseFormalityLevel": 4,
            "baseFriendlinessLevel": 3,
            "baseEmotionLevel": 3,
            "baseDirectnessLevel": 3,
            "negativePreferences": {
                "avoidFloweryLanguage": "moderate"
            }
        },
        "low_formal": {
            "baseFormalityLevel": 2,  # 캐주얼 (LoRA 사용 안됨)
            "baseFriendlinessLevel": 4,
            "baseEmotionLevel": 4,
            "baseDirectnessLevel": 3
        }
    }
    
    # 4. 기본 테스트 케이스 정의
    test_cases = [
        {
            "input": "안녕하세요! 회의 언제 하죠?",
            "context": "business",
            "description": "캐주얼한 업무 메시지 -> 공식 문서",
            "profile": "high_formal",
            "force_convert": False
        },
        {
            "input": "내일 프로젝트 끝내야 해요 😭 도와주세요",
            "context": "business", 
            "description": "감정적 표현 -> 공식적 요청서",
            "profile": "high_formal",
            "force_convert": False
        },
        {
            "input": "이거 왜 안되는지 모르겠네... 확인 좀 해주세요",
            "context": "report",
            "description": "불만 표현 -> 공식 보고서",
            "profile": "medium_formal",
            "force_convert": False
        },
        {
            "input": "급하게 처리해야 할 일이 있어서 연락드려요",
            "context": "business",
            "description": "급박한 상황 -> 정중한 요청",
            "profile": "high_formal",
            "force_convert": False
        },
        {
            "input": "좀 늦을 것 같아요 미안해요",
            "context": "personal",  # 개인적 -> LoRA 사용 안됨
            "description": "개인적 사과 -> 캐주얼 유지",
            "profile": "low_formal",
            "force_convert": False
        },
        # 강제 변환 테스트 케이스 추가
        {
            "input": "ㅋㅋ 그거 언제 끝나요? 😅",
            "context": "personal",
            "description": "이모티콘 포함 -> 사용자 요청으로 공식화",
            "profile": "low_formal",
            "force_convert": True
        },
        {
            "input": "아 진짜 짜증나네요 😤",
            "context": "personal", 
            "description": "감정 표현 -> 사용자 요청으로 공식화",
            "profile": "low_formal",
            "force_convert": True
        }
    ]
    
    # 5. 테스트 실행
    print(f"\n테스트 케이스 실행 ({len(test_cases)}개)")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {test_case['description']}")
        print(f"원본: {test_case['input']}")
        print(f"컨텍스트: {test_case['context']}")
        print(f"프로필: {test_case['profile']}")
        print(f"강제 변환: {test_case['force_convert']}")
        
        user_profile = test_user_profiles[test_case['profile']]
        
        try:
            # 공식 문서 변환 실행
            start_time = datetime.now()
            result = await finetune_chain.convert_to_formal(
                input_text=test_case['input'],
                user_profile=user_profile,
                context=test_case['context'],
                force_convert=test_case['force_convert']
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 결과 출력
            if result['success']:
                print(f"[성공] 변환 완료 ({duration:.2f}초)")
                print(f"변환 방법: {result['method']}")
                print(f"변환 이유: {result['reason']}")
                print(f"최종 결과: {result['converted_text']}")
                
                # LoRA 1차 변환 결과도 표시
                if result.get('lora_output') and result['lora_output'] != result['converted_text']:
                    print(f"LoRA 1차 변환: {result['lora_output']}")
                
            else:
                print(f"[실패] 변환 실패: {result['error']}")
                print(f"사용된 방법: {result['method']}")
                print(f"실패 이유: {result['reason']}")
                
        except Exception as e:
            print(f"[오류] 테스트 실행 오류: {e}")
        
        print("-" * 60)
    
    # 6. 조건별 변환 테스트
    print("\n조건별 변환 테스트")
    print("=" * 40)
    
    condition_tests = [
        {
            "name": "격식도 5 + business (무조건 True)",
            "profile": {"baseFormalityLevel": 5},
            "context": "business",
            "should_use_lora": True
        },
        {
            "name": "격식도 4 + business (True)", 
            "profile": {"baseFormalityLevel": 4},
            "context": "business",
            "should_use_lora": True
        },
        {
            "name": "격식도 4 + personal (False)",
            "profile": {"baseFormalityLevel": 4},
            "context": "personal", 
            "should_use_lora": False
        },
        {
            "name": "격식도 3 + business (False)",
            "profile": {"baseFormalityLevel": 3},
            "context": "business",
            "should_use_lora": False
        },
        {
            "name": "빈 프로필 + business (False)",
            "profile": {},
            "context": "business",
            "should_use_lora": False
        },
        {
            "name": "formal_document_mode = True",
            "profile": {"baseFormalityLevel": 2, "formal_document_mode": True},
            "context": "personal",
            "should_use_lora": True
        }
    ]
    
    for test in condition_tests:
        should_use = finetune_chain._should_use_lora(test['profile'], test['context'])
        label = "PASS" if should_use == test['should_use_lora'] else "FAIL"
        print(f"[{label}] {test['name']}: LoRA usage {should_use} (expected: {test['should_use_lora']})")
    
    # 7. Performance and memory information
    print("\nPerformance Information")
    print("=" * 40)
    
    device = status.get('device', 'unknown')
    if device == 'cuda':
        # Query only local GPU information
        try:
            import torch
            print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory max: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU device: {torch.cuda.get_device_name()}")
        except Exception as e:
            print(f"[WARNING] Failed to query GPU information: {e}")
    else:
        print("Running in CPU mode")
    
    print("\nFinetuneChain basic test completed!")

async def main():
    """Main test execution"""
    print("FinetuneChain Full Test Started!")
    print("=" * 80)
    
    # 1. Condition tests
    test_should_use_lora()
    
    # 2. Main integration test
    await test_finetune_chain()
    
    # 3. Force conversion test
    await test_force_convert()
    
    # 4. Convenience method test
    await test_convenience_method()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())