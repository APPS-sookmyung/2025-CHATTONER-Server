"""
RAG (Retrieval-Augmented Generation) Endpoints
Document-based Q&A and text quality analysis endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List, Annotated
from pydantic import BaseModel
from api.v1.schemas.conversion import UserProfile
from pathlib import Path

router = APIRouter()

# Request/Response Models
class DocumentIngestRequest(BaseModel):
    folder_path: str = "python_backend/langchain_pipeline/data/documents"

class DocumentIngestResponse(BaseModel):
    success: bool
    documents_processed: int
    message: str
    error: Optional[str] = None

class RAGQueryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    use_styles: bool = False
    user_profile: Optional[UserProfile] = None

class RAGQueryResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    converted_texts: Optional[Dict[str, str]] = None
    sources: List[Dict[str, Any]] = []
    rag_context: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class RAGStatusResponse(BaseModel):
    rag_status: str
    doc_count: int
    services_available: bool
    documents_path: str
    index_path: str

# RAG Service singleton
_rag_service_instance = None

def get_rag_service():
    """RAG service singleton instance"""
    global _rag_service_instance
    if _rag_service_instance is None:
        try:
            from services.rag_service import RAGService
            _rag_service_instance = RAGService()
        except ImportError as e:
            raise HTTPException(status_code=503, detail=f"RAG service is not available: {str(e)}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG service initialization failed: {str(e)}") from e
    return _rag_service_instance

@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_documents(
    request: DocumentIngestRequest,
    rag_service: Annotated[object, Depends(get_rag_service)]
) -> DocumentIngestResponse:
    """Create RAG vector DB from document folder"""
    try:
        folder_path = Path(request.folder_path)
        
        # Convert relative path to absolute path based on project root
        if not folder_path.is_absolute():
            project_root = Path(__file__).resolve().parents[4]  # 2025-CHATTONER-Server
            folder_path = project_root / folder_path
            
            
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Document folder not found: {request.folder_path} (resolved: {folder_path})")
        
        result = rag_service.ingest_documents(str(folder_path))
        
        return DocumentIngestResponse(
            success=result.get("success", False),
            documents_processed=result.get("documents_processed", 0),
            message="Document indexing completed." if result.get("success") else "Document indexing failed.",
            error=result.get("error") if not result.get("success") else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred during document indexing: {str(e)}")

@router.post("/ask", response_model=RAGQueryResponse)
async def ask_rag_question(
    request: RAGQueryRequest,
    rag_service: Annotated[object, Depends(get_rag_service)]
) -> RAGQueryResponse:
    """RAG-based Q&A"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Please enter a question.")
        
        if request.use_styles and request.user_profile:
            # 3-style conversion
            result = await rag_service.ask_with_styles(
                query=request.query,
                user_profile=(request.user_profile.model_dump() if request.user_profile else None),
                context=request.context or "personal"
            )
            
            return RAGQueryResponse(
                success=result.get("success", False),
                converted_texts=result.get("converted_texts", {}),
                sources=result.get("sources", []),
                rag_context=result.get("rag_context"),
                error=result.get("error") if not result.get("success") else None,
                metadata=result.get("metadata", {})
            )
        else:
            # Single answer
            result = await rag_service.ask_question(
                query=request.query,
                context=request.context
            )
            
            return RAGQueryResponse(
                success=result.get("success", False),
                answer=result.get("answer"),
                sources=result.get("sources", []),
                error=result.get("error") if not result.get("success") else None,
                metadata=result.get("metadata", {})
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred during Q&A: {str(e)}")

@router.get("/status", response_model=RAGStatusResponse)
async def get_rag_status(rag_service: Annotated[object, Depends(get_rag_service)]) -> RAGStatusResponse:
    """Check RAG system status"""
    try:
        status = rag_service.get_status()
        
        return RAGStatusResponse(
            rag_status=status.get("rag_status", "unknown"),
            doc_count=status.get("doc_count", 0),
            services_available=status.get("services_available", False),
            documents_path="python_backend/langchain_pipeline/data/documents",
            index_path="python_backend/langchain_pipeline/data/faiss_index"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred while checking status: {str(e)}")

@router.post("/analyze-grammar")
async def analyze_text_grammar(
    request: RAGQueryRequest,
    rag_service: Annotated[object, Depends(get_rag_service)]
) -> RAGQueryResponse:
    """RAG-based grammar analysis (document-based instead of GPT)"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Please enter text to analyze.")
        
        # Construct special query for grammar analysis
        grammar_query = f"Please analyze the grammar, spelling, and expression of the following text and provide improvement suggestions: {request.query}"
        
        result = await rag_service.ask_question(
            query=grammar_query,
            context="grammar analysis"
        )
        
        return RAGQueryResponse(
            success=result.get("success", False),
            answer=result.get("answer"),
            sources=result.get("sources", []),
            error=result.get("error") if not result.get("success") else None,
            metadata={
                **result.get("metadata", {}),
                "analysis_type": "grammar_check",
                "original_text": request.query
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred during grammar analysis: {str(e)}")

@router.post("/suggest-expressions")
async def suggest_better_expressions(
    request: RAGQueryRequest,
    rag_service: Annotated[object, Depends(get_rag_service)]
) -> RAGQueryResponse:
    """RAG-based expression improvement suggestions"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Please enter text to improve.")
        
        # Construct special query for expression improvement
        context_type = request.context or "business"
        improvement_query = f"Please change the following text to better expressions in {context_type} context: {request.query}"
        
        result = await rag_service.ask_question(
            query=improvement_query,
            context=f"{context_type} expression improvement"
        )
        
        return RAGQueryResponse(
            success=result.get("success", False),
            answer=result.get("answer"),
            sources=result.get("sources", []),
            error=result.get("error") if not result.get("success") else None,
            metadata={
                **result.get("metadata", {}),
                "analysis_type": "expression_improvement",
                "context_type": context_type,
                "original_text": request.query
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred during expression improvement suggestion: {str(e)}")