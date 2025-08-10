"""
FinetuneChain 테스트 스크립트 - force_convert 플래그 포함 버전
LoRA 파인튜닝 모델과 ChatGPT 2단계 변환 파이프라인 테스트

실행 방법:
python test_finetune_chain.py
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 경로 설정
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from langchain_pipeline.chains.finetune_chain import FinetuneChain

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_should_use_lora():
    """LoRA 사용 조건 테스트 (수정된 로직)"""
    print("\nLoRA 사용 조건 단위 테스트")
    print("=" * 40)
    
    try:
        from finetune_chain import FinetuneChain
        chain = FinetuneChain.__new__(FinetuneChain)
        
        test_cases = [
            # 격식도 5 이상 - 무조건 True
            ({"baseFormalityLevel": 5}, "personal", True),
            ({"baseFormalityLevel": 5}, "business", True),
            
            # 격식도 4 + business/report - True
            ({"baseFormalityLevel": 4}, "business", True),
            ({"baseFormalityLevel": 4}, "report", True),
            ({"baseFormalityLevel": 4}, "personal", False),  # personal은 안됨
            
            # 격식도 3 - business/report여도 False
            ({"baseFormalityLevel": 3}, "business", False),
            ({"baseFormalityLevel": 3}, "report", False),
            ({"baseFormalityLevel": 3}, "personal", False),
            
            # formal_document_mode = True - 무조건 True
            ({"baseFormalityLevel": 2, "formal_document_mode": True}, "personal", True),
            ({"baseFormalityLevel": 1, "formal_document_mode": True}, "business", True),
            
            # sessionFormalityLevel 우선 적용
            ({"sessionFormalityLevel": 5, "baseFormalityLevel": 2}, "business", True),
            ({"sessionFormalityLevel": 3, "baseFormalityLevel": 5}, "business", False),
            
            # 빈 프로필 (기본값 3) - False
            ({}, "business", False),
            ({}, "report", False),
            ({}, "personal", False),
        ]
        
        for profile, context, expected in test_cases:
            result = chain._should_use_lora(profile, context)
            status = "통과" if result == expected else "실패"
            print(f"[{status}] 프로필: {profile}, 컨텍스트: {context} -> {result} (예상: {expected})")
            
    except Exception as e:
        print(f"[오류] 단위 테스트 실패: {e}")

async def test_force_convert():
    """사용자 명시적 요청 테스트"""
    print("\n사용자 명시적 요청 테스트")
    print("=" * 60)
    
    try:
        finetune_chain = FinetuneChain()
    except Exception as e:
        print(f"[오류] FinetuneChain 초기화 실패: {e}")
        return
    
    # 원래는 변환 조건에 안 맞는 케이스들
    force_test_cases = [
        {
            "input": "안녕! 오늘 뭐해?",
            "profile": {"baseFormalityLevel": 1},  
            "context": "personal",  
            "description": "캐주얼 + 개인적 대화 (원래는 변환 안됨)"
        },
        {
            "input": "ㅋㅋㅋ 재밌네 ㅎㅎ 😂",
            "profile": {"baseFormalityLevel": 2},
            "context": "casual",
            "description": "이모티콘 포함 캐주얼 메시지"
        },
        {
            "input": "빨리빨리 해주세요!!",
            "profile": {},  
            "context": "personal",
            "description": "빈 프로필 + 급한 요청"
        }
    ]
    
    for i, test_case in enumerate(force_test_cases, 1):
        print(f"\n[강제변환 테스트 {i}] {test_case['description']}")
        print(f"원본: {test_case['input']}")
        
        # 1. 일반 변환 (force_convert=False) - 실패 예상
        try:
            result_normal = await finetune_chain.convert_to_formal(
                input_text=test_case['input'],
                user_profile=test_case['profile'],
                context=test_case['context'],
                force_convert=False
            )
            
            if result_normal['success']:
                print(f"일반 변환: [성공] (예상치 못함)")
                print(f"   결과: {result_normal['converted_text']}")
            else:
                print(f"일반 변환: [실패] {result_normal['error']}")
        except Exception as e:
            print(f"일반 변환: [오류] {e}")
        
        # 2. 강제 변환 (force_convert=True) - 성공 예상
        try:
            result_forced = await finetune_chain.convert_to_formal(
                input_text=test_case['input'],
                user_profile=test_case['profile'],
                context=test_case['context'],
                force_convert=True
            )
            
            if result_forced['success']:
                print(f"강제 변환: [성공] ({result_forced['method']})")
                print(f"   결과: {result_forced['converted_text']}")
                print(f"   변환 이유: {result_forced['reason']}")
                print(f"   강제 모드: {result_forced['forced']}")
            else:
                print(f"강제 변환: [실패] {result_forced['error']}")
        except Exception as e:
            print(f"강제 변환: [오류] {e}")
        
        print("-" * 60)

async def test_convenience_method():
    """편의 메서드 테스트"""
    print(f"\n편의 메서드 테스트")
    print("=" * 40)
    
    try:
        finetune_chain = FinetuneChain()
    except Exception as e:
        print(f"[오류] FinetuneChain 초기화 실패: {e}")
        return
    
    test_input = "야 이거 언제 하냐?"
    test_profile = {"baseFormalityLevel": 1}
    
    try:
        # convert_by_user_request 메서드 테스트
        result = await finetune_chain.convert_by_user_request(
            input_text=test_input,
            user_profile=test_profile,
            context="personal"
        )
        
        if result['success']:
            print(f"[성공] 편의 메서드")
            print(f"원본: {test_input}")
            print(f"변환: {result['converted_text']}")
            print(f"강제 변환: {result['forced']}")
            print(f"변환 이유: {result['reason']}")
            print(f"사용 방법: {result['method']}")
        else:
            print(f"[실패] 편의 메서드: {result['error']}")
    except Exception as e:
        print(f"[오류] 편의 메서드: {e}")

async def test_finetune_chain():
    """FinetuneChain 통합 테스트"""
    print("=" * 60)
    print("FinetuneChain 통합 테스트 시작")
    print("=" * 60)
    
    # 1. FinetuneChain 초기화
    try:
        print("\nFinetuneChain 초기화 중...")
        finetune_chain = FinetuneChain()
        print("[성공] FinetuneChain 초기화 완료")
    except Exception as e:
        print(f"[오류] FinetuneChain 초기화 실패: {e}")
        return
    
    # 2. 시스템 상태 확인
    print("\n시스템 상태 확인")
    print("=" * 40)
    status = finetune_chain.get_status()
    
    print(f"LoRA 상태: {status['lora_status']}")
    print(f"Services 사용 가능: {status['services_available']}")
    print(f"기본 모델 로드됨: {status['base_model_loaded']}")
    print(f"디바이스: {status['device']}")
    print(f"LoRA 모델 경로: {status['lora_model_path']}")
    print(f"모델명: {status.get('model_name', 'N/A')}")
    
    # 3. 테스트용 사용자 프로필 설정
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
    print(f"\n조건별 변환 테스트")
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
        status = "통과" if should_use == test['should_use_lora'] else "실패"
        print(f"[{status}] {test['name']}: LoRA 사용 {should_use} (예상: {test['should_use_lora']})")
    
    # 7. 성능 및 메모리 정보
    print(f"\n성능 정보")
    print("=" * 40)
    
    if status['device'] == 'cuda':
        try:
            import torch
            print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU 메모리 최대: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU 디바이스: {torch.cuda.get_device_name()}")
        except Exception as e:
            print(f"[경고] GPU 정보 조회 실패: {e}")
    else:
        print("CPU 모드로 실행 중")
    
    print(f"\nFinetuneChain 기본 테스트 완료!")

async def main():
    """메인 테스트 실행"""
    print("FinetuneChain 전체 테스트 시작!")
    print("=" * 80)
    
    # 1. 조건 테스트
    test_should_use_lora()
    
    # 2. 메인 통합 테스트
    await test_finetune_chain()
    
    # 3. 강제 변환 테스트
    await test_force_convert()
    
    # 4. 편의 메서드 테스트
    await test_convenience_method()
    
    print("\n" + "=" * 80)
    print("모든 테스트 완료!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())