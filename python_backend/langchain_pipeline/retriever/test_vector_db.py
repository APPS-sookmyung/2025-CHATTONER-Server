"""
Vector DB 테스트 파일
- 문서 인덱싱 테스트
- FAISS 로드 테스트
- 검색 기능 테스트
"""

import logging
from pathlib import Path
from vector_db import ingest_documents_from_folder, FAISS_INDEX_PATH, DOCS_PATH, embedding
from langchain_community.vectorstores import FAISS

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_document_ingestion():
    """문서 인덱싱 테스트"""
    logger.info(f"문서 폴더: {DOCS_PATH}")
    logger.info(f"폴더 존재 여부: {DOCS_PATH.exists()}")
    
    if not DOCS_PATH.exists():
        logger.error("문서 폴더가 존재하지 않습니다")
        return False
    
    # 문서 인덱싱 실행
    result = ingest_documents_from_folder(DOCS_PATH)
    
    if result and result[0] is not None:
        vectorstore, all_docs = result
        logger.info(f"인덱싱 완료: {len(all_docs)}개 청크")
        return True
    else:
        logger.error("문서 인덱싱 실패")
        return False


def test_faiss_loading():
    """FAISS 인덱스 로드 및 검색 테스트"""
    logger.info(f"인덱스 경로: {FAISS_INDEX_PATH}")
    logger.info(f"인덱스 존재 여부: {FAISS_INDEX_PATH.exists()}")
    
    if not FAISS_INDEX_PATH.exists():
        logger.error("FAISS 인덱스가 존재하지 않습니다")
        return False
    
    try:
        # FAISS 로드
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        
        logger.info(f"FAISS 로드 성공 - 문서 수: {vectorstore.index.ntotal}, 차원: {vectorstore.index.d}")
        
        # 검색 테스트
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke("LangChain")
        
        logger.info(f"검색 테스트 완료 - 검색된 문서: {len(docs)}개")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            logger.info(f"  문서 {i}: {source} - {content_preview}")
        
        return True
        
    except Exception as e:
        logger.error(f"FAISS 로드 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    logger.info("Vector DB 테스트 시작")
    
    # 기존 인덱스가 있으면 로드 테스트, 없으면 인덱싱부터
    if FAISS_INDEX_PATH.exists():
        logger.info("기존 인덱스 발견 - 로드 테스트 실행")
        success = test_faiss_loading()
    else:
        logger.info("기존 인덱스 없음 - 문서 인덱싱부터 실행")
        success = test_document_ingestion()
        
        if success:
            success = test_faiss_loading()
    
    if success:
        logger.info("모든 테스트 통과!")
    else:
        logger.error("테스트 실패")


if __name__ == "__main__":
    main()