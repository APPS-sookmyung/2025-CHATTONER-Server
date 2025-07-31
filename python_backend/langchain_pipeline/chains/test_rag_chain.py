"""
RAG Chain 테스트 파일
- 기본 RAG 기능 테스트
- 3가지 스타일 변환 테스트
- Chunk 정보 확인
"""

import sys
from pathlib import Path
import asyncio


# 프로젝트 경로 설정
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from rag_chain import RAGChain

# 기본 기능 테스트
def test_basic_rag():
    rag = RAGChain()
    status = rag.get_status()
    print(f" RAG 상태: {status['rag_status']} ({status['doc_count']}개 문서)")
    
    if not rag.is_initialized:
        print("문서를 먼저 인덱싱해주세요")
        return
    
    test_queries = [
        "이 문서의 주요 내용은?",
        "핵심 키워드들을 나열해주세요", 
        "어떤 기술에 대한 내용인가요?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n 질문 {i}: {query}")
        result = rag.ask(query)
        
        if result["success"]:
            print(f" 답변: {result['answer'][:100]}...")
            print(f" 참조: {len(result['sources'])}개 문서")
        else:
            print(f" 오류: {result['answer']}")

# 문서 청크 정보 확인
def test_chunk_info():
    rag = RAGChain()
    if not rag.is_initialized:
        print(" 문서가 인덱싱되지 않았습니다")
        return
    
    query = "LangChain"
    docs = rag.retriever.get_relevant_documents(query)
    
    print(f" '{query}' 검색 결과:")
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        metadata = doc.metadata
        print(f"  Chunk {i}:")
        print(f"  길이: {len(content)}자")
        print(f"  소스: {metadata.get('source', 'Unknown')}")
        print(f"  미리보기: {content[:80]}...")
        print()


async def test_styles_rag():
    
    rag = RAGChain()
    
    if not rag.services_available:
        print(" Services가 로드되지 않아 3가지 스타일 테스트를 건너뜁니다")
        return
    
    if not rag.is_initialized:
        print(" 문서가 인덱싱되지 않았습니다")
        return
    
    # 테스트용 사용자 프로필
    test_profile = {
        "baseFormalityLevel": 3,
        "baseFriendlinessLevel": 4,
        "baseEmotionLevel": 3,
        "baseDirectnessLevel": 3
    }
    
    test_query = "LangChain이 뭔가요?"
    print(f"\n 질문: {test_query}")
    
    try:
        result = await rag.ask_with_styles(test_query, test_profile, "personal")
        
        if result["success"]:
            print("\n 3가지 스타일 답변:")
            converted_texts = result.get("converted_texts", {})
            
            for style, answer in converted_texts.items():
                print(f"\n {style.upper()} 스타일:")
                # 답변 길이 제한해서 출력
                display_answer = answer[:150] + "..." if len(answer) > 150 else answer
                print(f"   {display_answer}")
            
            if "sources" in result:
                print(f"\n 참조 문서: {len(result['sources'])}개")
            
            if "rag_context" in result:
                context_preview = result['rag_context'][:100] + "..." if len(result['rag_context']) > 100 else result['rag_context']
                print(f"\n RAG 컨텍스트 미리보기:")
                print(f"   {context_preview}")
                
        else:
            print(f" 스타일 변환 오류: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f" Async 테스트 오류: {e}")

# 빠른 질의 테스트
def test_quick_functions():
    rag = RAGChain()
    
    print("\n 빠른 질문 테스트:")
    if rag.is_initialized:
        result = rag.ask("FAISS가 뭔가요?")
        if result["success"]:
            print(f"답변: {result['answer'][:100]}...")
        else:
            print(f"오류: {result['answer']}")
    else:
        print("문서가 인덱싱되지 않았습니다")
    


def main():
    """메인 테스트 실행"""
    print(" RAG Chain 통합 테스트 시작\n")
    
    # 1. RAG Chain 상태 확인
    rag = RAGChain()
    status = rag.get_status()
    print(f" RAG 상태: {status['rag_status']} ({status['doc_count']}개 문서)")
    print(f" Services 상태: {'사용 가능' if status['services_available'] else '사용 불가'}")
    
    # 문서가 없으면 테스트 중단
    if status["rag_status"] == "not_ready":
        print("\n 문서 인덱싱이 필요합니다")
        return
    
    print("\n" + "=" * 60)
    
    # 2. 테스트 실행
    test_chunk_info()
    test_basic_rag()
    test_quick_functions()
    
    # 3. 비동기 테스트 (Services가 있는 경우만)
    if status['services_available']:
        asyncio.run(test_styles_rag())
    else:
        print("\n Services가 없어 3가지 스타일 테스트를 건너뜁니다")
    
    print("\n 모든 테스트 완료!")


if __name__ == "__main__":
    main()