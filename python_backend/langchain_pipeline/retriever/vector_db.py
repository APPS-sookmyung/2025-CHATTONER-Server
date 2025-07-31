"""
지정한 폴더 내의 .txt 및 .pdf 파일을 불러와
문서를 청크 단위로 분할하고 HuggingFace 임베딩을 적용한 뒤,
FAISS 벡터 저장소로 저장합니다.

Args:
    folder_path (Path): 문서가 위치한 폴더 경로

Returns:
    Tuple[FAISS, List[Document]]: 생성된 벡터스토어와 청크 문서 리스트
"""

import sys
import logging
from pathlib import Path

# 프로젝트 경로 설정
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_pipeline.retriever.doc_splitter import split_documents

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 로깅 설정
logger = logging.getLogger(__name__)

# HuggingFace 임베딩 모델 설정
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# 경로 설정
script_dir = Path(__file__).parent
FAISS_INDEX_PATH = script_dir / "index" / "faiss_index"
DOCS_PATH = script_dir / "index" / "docs"


def ingest_documents_from_folder(folder_path: Path):
    all_docs = []
    
    if not folder_path.exists():
        logger.error(f"폴더가 존재하지 않습니다: {folder_path}")
        return None, []

    # 폴더 내 모든 파일 처리
    for filepath in folder_path.glob("*"):
        if filepath.suffix == ".txt":
            loader = TextLoader(str(filepath), encoding="utf-8")
        elif filepath.suffix == ".pdf":
            loader = PyPDFLoader(str(filepath))
        else:
            continue #그 외 확정자 무시
        
        # 파일 로딩 및 청크 분할
        try:
            raw_docs = loader.load()
            chunked_docs = split_documents(raw_docs, chunk_size=500, chunk_overlap=50)
            
            
            for doc in chunked_docs:
                doc.metadata["source"] = filepath.name

            all_docs.extend(chunked_docs)
            logger.info(f"처리 완료: {filepath.name} ({len(chunked_docs)} 청크)")
            
        except Exception as e:
            logger.error(f"파일 로딩 실패 ({filepath.name}): {e}")
            continue

    if not all_docs:
        logger.warning("처리할 문서가 없습니다.")
        return None, []
    
    # 벡터 저장소 생성 및 저장
    try:
        vectorstore = FAISS.from_documents(all_docs, embedding)
        vectorstore.save_local(str(FAISS_INDEX_PATH))
        logger.info(f"벡터 인덱스 저장 완료: {len(all_docs)}개 청크, 경로: {FAISS_INDEX_PATH}")
        return vectorstore, all_docs
    except Exception as e:
        logger.error(f"벡터 저장 실패: {e}")
        return None, all_docs