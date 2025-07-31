# run_ingest.py
#/from langchain_pipeline.retriever.vector_db import ingest_documents_from_folder

#if __name__ == "__main__":

from langchain_pipeline.retriever.vector_db import ingest_documents_from_folder

if __name__ == "__main__":
    ingest_documents_from_folder("langchain_pipeline/retriever/index/faiss_index")
