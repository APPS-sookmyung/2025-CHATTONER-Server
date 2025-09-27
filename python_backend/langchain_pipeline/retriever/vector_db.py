"""
FAISS 벡터 데이터베이스 관리
문서 임베딩, 저장, 검색 기능 제공
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# 프로젝트 루트 경로 설정
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# LangChain imports
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain.schema import Document
    from dotenv import load_dotenv
except ImportError as e:
    logging.error(f"LangChain 라이브러리 import 실패: {e}")
    raise

# 로거 설정
logger = logging.getLogger(__name__)

# 설정 (공통 설정에서 가져옴)
def _get_config_paths():
    """설정 경로 반환"""
    from core.rag_config import get_rag_config
    config = get_rag_config()
    return config.faiss_index_path, config.documents_path

FAISS_INDEX_PATH, DOCUMENTS_PATH = _get_config_paths()

def _is_trusted_path(path: Path) -> bool:
    """경로가 신뢰할 수 있는지 검증"""
    try:
        # 절대 경로로 변환
        abs_path = path.resolve()

        # 프로젝트 루트 경로
        project_root = Path(__file__).resolve().parents[3]

        # 프로젝트 내부 경로인지 확인
        try:
            abs_path.relative_to(project_root)
        except ValueError:
            logger.warning(f"프로젝트 외부 경로 접근 시도: {abs_path}")
            return False

        # 허용된 디렉토리 목록
        allowed_dirs = [
            "python_backend/langchain_pipeline/data",
            "data/faiss_index",
            "langchain_pipeline/data"
        ]

        for allowed_dir in allowed_dirs:
            allowed_path = project_root / allowed_dir
            try:
                abs_path.relative_to(allowed_path.resolve())
                return True
            except ValueError:
                continue

        logger.warning(f"허용되지 않은 디렉토리: {abs_path}")
        return False

    except Exception as e:
        logger.error(f"경로 검증 중 오류: {e}")
        return False

def _is_project_managed_path(path: Path) -> bool:
    """프로젝트에서 직접 관리하는 경로인지 확인"""
    try:
        abs_path = path.resolve()
        project_root = Path(__file__).resolve().parents[3]

        # 기본 FAISS 인덱스 경로와 비교
        default_index_path = project_root / FAISS_INDEX_PATH

        return abs_path == default_index_path.resolve()

    except Exception as e:
        logger.error(f"프로젝트 관리 경로 확인 중 오류: {e}")
        return False

def get_embedding():
    """OpenAI 임베딩 인스턴스 반환 (공통 설정 사용)"""
    from core.rag_config import get_rag_config

    config = get_rag_config()
    api_key = config.get_openai_api_key()

    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")

    return OpenAIEmbeddings(
        model=config.get_embedding_model(),
        api_key=api_key
    )

def load_documents_from_folder(folder_path: Path) -> List[Document]:
    """폴더에서 문서 로드"""
    documents = []
    
    if not folder_path.exists():
        logger.warning(f"문서 폴더가 존재하지 않음: {folder_path}")
        return documents
    
    # .txt 파일 로드
    for txt_file in folder_path.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.strip():
                doc = Document(
                    page_content=content,
                    metadata={"source": str(txt_file.name)}
                )
                documents.append(doc)
                logger.info(f"텍스트 파일 로드: {txt_file.name}")
                
        except Exception as e:
            logger.error(f"텍스트 파일 로드 실패 {txt_file}: {e}")
    
    # .pdf 파일 로드 (있는 경우)
    for pdf_file in folder_path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            pdf_docs = loader.load()
            
            for doc in pdf_docs:
                doc.metadata["source"] = pdf_file.name
            
            documents.extend(pdf_docs)
            logger.info(f"PDF 파일 로드: {pdf_file.name} ({len(pdf_docs)}페이지)")
            
        except Exception as e:
            logger.error(f"PDF 파일 로드 실패 {pdf_file}: {e}")
    
    logger.info(f"총 {len(documents)}개 문서 로드됨")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """문서를 청크로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"문서 분할 완료: {len(documents)}개 → {len(split_docs)}개 청크")
    
    return split_docs

def create_vector_store(documents: List[Document]) -> FAISS:
    """문서로부터 FAISS 벡터 저장소 생성"""
    try:
        embeddings = get_embedding()
        
        if not documents:
            raise ValueError("임베딩할 문서가 없습니다")
        
        # FAISS 벡터 저장소 생성
        vectorstore = FAISS.from_documents(documents, embeddings)
        logger.info(f"FAISS 벡터 저장소 생성 완료: {len(documents)}개 문서")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"벡터 저장소 생성 실패: {e}")
        raise

def save_vector_store(vectorstore: FAISS, save_path: Path):
    """벡터 저장소를 파일로 저장"""
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        
        vectorstore.save_local(str(save_path))
        logger.info(f"벡터 저장소 저장됨: {save_path}")
        
    except Exception as e:
        logger.error(f"벡터 저장소 저장 실패: {e}")
        raise

def _is_trusted_index_path(path: Path) -> bool:
    """신뢰 가능한 인덱스 경로인지 확인 (프로젝트 관리 디렉터리 내 고정)."""
    try:
        return path.resolve() == FAISS_INDEX_PATH.resolve()
    except Exception:
        return False


def load_vector_store(load_path: Path) -> Optional[FAISS]:
    """저장된 벡터 저장소 로드 (보안 강화)"""
    try:
        if not load_path.exists() or not any(load_path.iterdir()):
            logger.warning(f"벡터 저장소가 없음: {load_path}")
            return None

        # 보안 검증: 신뢰할 수 있는 경로에서만 로드
        if not _is_trusted_path(load_path):
            logger.error(f"신뢰할 수 없는 경로에서 벡터 저장소 로드 시도: {load_path}")
            return None

        # 필수 FAISS 파일 존재 여부 검증
        required_files = ["index.faiss", "index.pkl"]
        for file_name in required_files:
            file_path = load_path / file_name
            if not file_path.exists():
                logger.error(f"필수 FAISS 파일이 없음: {file_path}")
                return None

            # 파일 크기 검증 (비정상적으로 큰 파일 방지)
            if file_path.stat().st_size > 500 * 1024 * 1024:  # 500MB 제한
                logger.error(f"FAISS 파일이 너무 큼: {file_path}")
                return None

        embeddings = get_embedding()

        # 보안 강화: allow_dangerous_deserialization=False로 변경
        # 대신 자체 검증된 경로만 허용
        try:
            vectorstore = FAISS.load_local(
                str(load_path),
                embeddings,
                allow_dangerous_deserialization=False
            )
        except Exception as security_error:
            logger.warning(f"안전한 역직렬화 실패, 신뢰된 경로 재시도: {security_error}")
            # 신뢰된 경로에서만 위험한 역직렬화 허용
            if _is_project_managed_path(load_path):
                vectorstore = FAISS.load_local(
                    str(load_path),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.warning("프로젝트 관리 경로에서 위험한 역직렬화 사용됨")
            else:
                raise security_error

        logger.info(f"벡터 저장소 안전하게 로드됨: {vectorstore.index.ntotal}개 벡터")
        return vectorstore

    except Exception as e:
        logger.exception(f"벡터 저장소 로드 실패: {e}")
        return None

def ingest_documents_from_folder(folder_path: Path) -> Tuple[Optional[FAISS], List[Document]]:
    """
    폴더에서 문서를 로드하고 벡터 저장소 생성 (PostgreSQL 메타데이터 포함)

    Args:
        folder_path: 문서가 있는 폴더 경로

    Returns:
        (벡터저장소, 원본문서리스트) 튜플
    """
    try:
        logger.info(f"문서 인덱싱 시작: {folder_path}")

        # 1. 문서 로드
        documents = load_documents_from_folder(folder_path)
        if not documents:
            logger.warning("로드된 문서가 없습니다")
            return None, []

        # 2. 문서 분할
        split_docs = split_documents(documents)

        # 3. 벡터 저장소 생성
        vectorstore = create_vector_store(split_docs)

        # 4. 저장
        save_vector_store(vectorstore, FAISS_INDEX_PATH)

        # 5. PostgreSQL에 메타데이터 저장
        _save_document_metadata_to_postgres(folder_path, documents, split_docs, vectorstore)

        logger.info("문서 인덱싱 완료")
        return vectorstore, documents

    except Exception as e:
        logger.exception(f"문서 인덱싱 실패: {e}")
        return None, []

def _save_document_metadata_to_postgres(folder_path: Path, documents: List[Document],
                                       split_docs: List[Document], vectorstore: FAISS):
    """PostgreSQL에 벡터 문서 메타데이터 저장"""
    try:
        from database.storage import DatabaseStorage
        from core.rag_config import get_rag_config

        config = get_rag_config()
        storage = DatabaseStorage()

        for doc in documents:
            file_path = doc.metadata.get("source", "unknown")
            file_path_obj = folder_path / file_path if not Path(file_path).is_absolute() else Path(file_path)

            if file_path_obj.exists():
                file_size = file_path_obj.stat().st_size

                # 메타데이터 구성
                metadata = {
                    "file_name": file_path_obj.name,
                    "file_path": str(file_path_obj),
                    "file_size_bytes": file_size,
                    "content_type": _get_content_type(file_path_obj),
                    "embedding_model": config.get_embedding_model(),
                    "chunk_count": len([d for d in split_docs if d.metadata.get("source") == file_path]),
                    "chunk_size": config.get_chunk_size(),
                    "chunk_overlap": config.get_chunk_overlap(),
                    "faiss_index_path": str(FAISS_INDEX_PATH),
                    "vector_dimension": vectorstore.index.d if hasattr(vectorstore.index, 'd') else 1536,
                    "status": "active"
                }

                success = storage.save_vector_document_metadata(metadata)
                if success:
                    logger.info(f"문서 메타데이터 저장 완료: {file_path_obj.name}")
                else:
                    logger.warning(f"문서 메타데이터 저장 실패: {file_path_obj.name}")

    except Exception as e:
        logger.error(f"PostgreSQL 메타데이터 저장 중 오류: {e}")

def _get_content_type(file_path: Path) -> str:
    """파일 확장자에 따른 컨텐츠 타입 반환"""
    suffix = file_path.suffix.lower()
    content_types = {
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".md": "text/markdown",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
    return content_types.get(suffix, "text/plain")

def search_similar_documents(query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
    """유사한 문서 검색"""
    try:
        vectorstore = load_vector_store(FAISS_INDEX_PATH)
        if not vectorstore:
            logger.warning("벡터 저장소가 로드되지 않았습니다")
            return []
        
        # 유사도 검색
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        
        logger.info(f"검색 완료: {len(results)}개 결과")
        return results
        
    except Exception as e:
        logger.exception(f"문서 검색 실패: {e}")
        return []

def get_vector_store_stats() -> Dict[str, Any]:
    """벡터 저장소 통계 정보"""
    try:
        vectorstore = load_vector_store(FAISS_INDEX_PATH)
        if not vectorstore:
            return {
                "status": "not_found",
                "document_count": 0,
                "index_path": str(FAISS_INDEX_PATH)
            }
        
        return {
            "status": "ready",
            "document_count": vectorstore.index.ntotal,
            "index_path": str(FAISS_INDEX_PATH),
            "documents_path": str(DOCUMENTS_PATH)
        }
        
    except Exception as e:
        logger.exception(f"통계 조회 실패: {e}")
        return {
            "status": "error",
            "error": str(e),
            "document_count": 0
        }

if __name__ == "__main__":
    # 테스트 코드
    print("🔍 FAISS 벡터 데이터베이스 테스트")
    
    # 문서 인덱싱
    vectorstore, docs = ingest_documents_from_folder(DOCUMENTS_PATH)
    
    if vectorstore:
        print(f"✅ 인덱싱 완료: {len(docs)}개 문서")
        
        # 검색 테스트
        test_query = "비즈니스 메일 작성"
        results = search_similar_documents(test_query, top_k=3)
        
        print(f"\n🔍 검색 결과 ('{test_query}'):")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. 점수: {score:.3f}")
            print(f"   내용: {doc.page_content[:100]}...")
            print(f"   출처: {doc.metadata.get('source', 'Unknown')}")
            print()
    else:
        print("❌ 인덱싱 실패")
