"""
RAG Chain Test File
- Basic RAG functionality test
- 3-style conversion test
- Chunk information check
"""

import sys
from pathlib import Path
import asyncio


# Project path configuration
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from rag_chain import RAGChain

# Basic functionality test
def test_basic_rag():
    rag = RAGChain()
    status = rag.get_status()
    print(f" RAG status: {status['rag_status']} ({status['doc_count']} documents)")
    
    if not rag.is_initialized:
        print("Please index documents first")
        return
    
    test_queries = [
        "이 문서의 주요 내용은?",
        "핵심 키워드들을 나열해주세요", 
        "어떤 기술에 대한 내용인가요?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n Question {i}: {query}")
        result = rag.ask(query)
        
        if result["success"]:
            print(f" Answer: {result['answer'][:100]}...")
            print(f" References: {len(result['sources'])} documents")
        else:
            print(f" Error: {result['answer']}")

# Document chunk information check
def test_chunk_info():
    rag = RAGChain()
    if not rag.is_initialized:
        print(" Documents are not indexed")
        return
    
    query = "LangChain"
    docs = rag.retriever.get_relevant_documents(query)
    
    print(f" '{query}' search results:")
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        metadata = doc.metadata
        print(f"  Chunk {i}:")
        print(f"  Length: {len(content)} characters")
        print(f"  Source: {metadata.get('source', 'Unknown')}")
        print(f"  Preview: {content[:80]}...")
        print()


async def test_styles_rag():
    
    rag = RAGChain()
    
    if not rag.services_available:
        print(" Services are not loaded, skipping 3-style test")
        return
    
    if not rag.is_initialized:
        print(" Documents are not indexed")
        return
    
    # Test user profile
    test_profile = {
        "baseFormalityLevel": 3,
        "baseFriendlinessLevel": 4,
        "baseEmotionLevel": 3,
        "baseDirectnessLevel": 3
    }
    
    test_query = "LangChain이 뭔가요?"
    print(f"\n Question: {test_query}")
    
    try:
        result = await rag.ask_with_styles(test_query, test_profile, "personal")
        
        if result["success"]:
            print("\n 3-style answers:")
            converted_texts = result.get("converted_texts", {})
            
            for style, answer in converted_texts.items():
                print(f"\n {style.upper()} 스타일:")
                # Print with limited answer length
                display_answer = answer[:150] + "..." if len(answer) > 150 else answer
                print(f"   {display_answer}")
            
            if "sources" in result:
                print(f"\n Reference documents: {len(result['sources'])}")
            
            if "rag_context" in result:
                context_preview = result['rag_context'][:100] + "..." if len(result['rag_context']) > 100 else result['rag_context']
                print("\n RAG context preview:")
                print(f"   {context_preview}")
                
        else:
            print(f" Style conversion error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f" Async test error: {e}")

# Quick query test
def test_quick_functions():
    rag = RAGChain()
    
    print("\n Quick question test:")
    if rag.is_initialized:
        result = rag.ask("FAISS가 뭔가요?")
        if result["success"]:
            print(f"Answer: {result['answer'][:100]}...")
        else:
            print(f"Error: {result['answer']}")
    else:
        print("Documents are not indexed")
    


def main():
    """Main test execution"""
    print(" RAG Chain Integration Test Started\n")
    
    # 1. RAG Chain 상태 확인
    rag = RAGChain()
    status = rag.get_status()
    print(f" RAG status: {status['rag_status']} ({status['doc_count']} documents)")
    print(f" Services status: {'Available' if status['services_available'] else 'Unavailable'}")
    
    # 문서가 없으면 테스트 중단
    if status["rag_status"] == "not_ready":
        print("\n Document indexing is required")
        return
    
    print("\n" + "=" * 60)
    
    # 2. Execute tests
    test_chunk_info()
    test_basic_rag()
    test_quick_functions()
    
    # 3. Async test (only when Services are available)
    if status['services_available']:
        asyncio.run(test_styles_rag())
    else:
        print("\n Skipping 3-style test due to missing Services")
    
    print("\n All tests completed!")


if __name__ == "__main__":
    main()