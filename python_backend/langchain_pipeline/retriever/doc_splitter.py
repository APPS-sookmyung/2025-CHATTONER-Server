"""
LangChain의 RecursiveCharacterTextSplitter를 사용하여
문서를 청크 단위로 분할하는 유틸 함수 모듈입니다.

주요 함수:
- get_text_splitter(): Splitter 객체 생성
- split_documents(): Document 리스트를 청크 단위로 분할

Args:
        chunk_size (int): 각 청크의 최대 길이 (기본값: 500)
        chunk_overlap (int): 청크 간 중첩 길이 (기본값: 50)
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def get_text_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )



def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:

    splitter = get_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)
