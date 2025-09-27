"""
Index documents based on FAISS vector store
Provide document-based Q&A functionality through GPT-based LLM
Apply 3 versions of responses tailored to user's speaking style

Key Features:
- Load .txt and .pdf files from document folders and vectorize and store them
- Load stored vector index to perform search-based Q&A
- Provide style-converted responses according to user speaking profile
- Include system status and document index status checking functionality
"""

import sys
import logging
from pathlib import Path
import os
from typing import Dict, Optional, Any
from datetime import datetime

# Project path configuration
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from core.config import get_settings 
from langchain_pipeline.retriever.vector_db import ingest_documents_from_folder, FAISS_INDEX_PATH, get_embedding
settings = get_settings()
# Logger configuration
logger = logging.getLogger(__name__)

class RAGChain:
    """RAG (Retrieval-Augmented Generation) chain"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        """Initialize RAG Chain (using common configuration)"""
        from core.rag_config import get_rag_config

        # Load common configuration
        self.config = get_rag_config()

        # Services lazy loading (initialize only when needed)
        self.services_available = False
        self._services_cache = {}
        self._check_services_availability()

        api_key = self.config.get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        self.vectorstore = None
        self.retriever = None
        self.is_initialized = False
        
        # Default prompt
        self.default_rag_prompt = PromptTemplate.from_template("""
Please answer the question based on the following document content.

Document:
{context}

Question: {question}

Answer:
""")
        
        # Attempt automatic initialization
        self._load_vectorstore()
    
    def _check_services_availability(self):
        """Check Services availability (without actually loading)"""
        import importlib.util
        
        required_modules = [
            "services.prompt_engineering",
            "services.openai_services", 
            "services.conversion_service",
            "services.user_preferences"
        ]
        
        try:
            for module_name in required_modules:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.services_available = False
                    logger.warning(f"Services module not found: {module_name}")
                    return
            
            self.services_available = True
            logger.info("Services module availability check completed")
        except Exception as e:
            self.services_available = False
            logger.warning(f"Services module availability check failed: {e}")
    
    def _get_service(self, service_name: str):
        """Service lazy loading"""
        if service_name in self._services_cache:
            return self._services_cache[service_name]
        
        if not self.services_available:
            return None
        
        try:
            if service_name == "conversion_service":
                from services.conversion_service import ConversionService
                service = ConversionService()
            elif service_name == "user_preferences_service":
                from services.user_preferences import UserPreferencesService
                service = UserPreferencesService()
            else:
                return None
            
            self._services_cache[service_name] = service
            logger.info(f"{service_name} loading completed")
            return service
            
        except Exception as e:
            logger.error(f"{service_name} loading failed: {e}")
            return None
    
    def _load_vectorstore(self):
        """Load existing index (security enhanced)"""
        try:
            # Use safe load function from vector_db.py
            from langchain_pipeline.retriever.vector_db import load_vector_store

            self.vectorstore = load_vector_store(FAISS_INDEX_PATH)

            if self.vectorstore:
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                self.is_initialized = True
                logger.info(f"RAG Chain safely prepared: {self.vectorstore.index.ntotal} documents")
            else:
                logger.warning("Vector store loading failed - index is missing or untrustworthy")

        except Exception as e:
            logger.warning(f"Existing index loading failed: {e}")
    
    def ingest_documents(self, folder_path: str) -> Dict:
        """Create vector DB from document folder"""
        try:
            result = ingest_documents_from_folder(Path(folder_path))
            if result and result[0] is not None:
                self._load_vectorstore()  # Reload
                return {"success": True, "documents_processed": len(result[1])}
            return {"success": False, "error": "Document processing failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ask(self, query: str, context: Optional[str] = None) -> Dict:
        """Ask question"""
        if not self.is_initialized:
            return {"success": False, "answer": "Documents are not indexed."}
        
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.default_rag_prompt},
                return_source_documents=True
            )
            
            # Execute query
            if context and context.strip():
                enhanced_query = f"Context: {context.strip()}\n\nQuestion: {query}"
            else:
                enhanced_query = query

            result = qa_chain.invoke({"query": enhanced_query})
            
            # Organize results
            source_docs = result.get("source_documents", [])
            chunks = [
                {
                    "content": doc.page_content[:100] + "...",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in source_docs
            ]
            
            return {
                "success": True,
                "answer": result["result"],
                "sources": chunks,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error occurred: {e}",
                "sources": []
            }
    
    async def ask_with_styles(self, query: str, user_profile: Dict, context: str = "personal") -> Dict:
        """3-style RAG answers (improved version)"""
        error_response = {
            "success": False,
            "converted_texts": {"direct": "", "gentle": "", "neutral": ""},
            "sources": [],
            "metadata": {
                "query_timestamp": datetime.now().isoformat(),
                "model_used": "none"
            }
        }
        
        # ConversionService 지연 로딩
        conversion_service = self._get_service("conversion_service")
        if not conversion_service:
            error_response["error"] = "Cannot load ConversionService."
            return error_response
        
        if not self.is_initialized:
            error_response["error"] = "RAG system is not initialized."
            return error_response
        
        try:
            # Document search
            docs = self.retriever.get_relevant_documents(query)
            if not docs:
                error_response["error"] = "No related documents found."
                return error_response
            
            # Organize retrieved documents as context
            context_parts = []
            sources = []
            
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", f"Document_{i}")
                content = doc.page_content
                context_parts.append(f"[Reference Document {i}] ({source}):\n{content}")
                
                sources.append({
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "source": source,
                    "rank": i
                })
            
            retrieved_docs = "\n\n".join(context_parts)
            enhanced_input = f"Question: {query}\n\nReference Documents:\n{retrieved_docs}"
            
            # Style conversion through ConversionService
            result = await conversion_service.convert_text(
                input_text=enhanced_input,
                user_profile=user_profile,
                context=context
            )
            
            # Add RAG-related information
            if result.get("success"):
                result["sources"] = sources
                result["rag_context"] = retrieved_docs[:300] + "..." if len(retrieved_docs) > 300 else retrieved_docs
                result["metadata"]["documents_retrieved"] = len(docs)
                result["metadata"]["model_used"] = "gpt-4o + faiss"
            
            return result
            
        except Exception as e:
            logger.error(f"Error during style conversion: {e}")
            error_response["error"] = f"Error occurred during style conversion: {str(e)}"
            return error_response
        
    async def process_user_feedback(self,
                                   feedback_text: str,
                                   user_profile: Dict[str, Any],
                                   rating: Optional[int] = None,
                                   selected_version: str = "neutral") -> Dict[str, Any]:
        """User feedback processing - advanced/basic processing selection (improved version)"""
        try:
            # Attempt advanced processing (UserPreferencesService)
            if rating is not None:
                user_preferences_service = self._get_service("user_preferences_service")
                if user_preferences_service:
                    user_id = user_profile.get('userId', 'unknown')
                    success = await user_preferences_service.adapt_user_style(
                        user_id=user_id,
                        feedback_text=feedback_text,
                        rating=rating,
                        selected_version=selected_version
                    )
                    
                    if success:
                        logger.info(f"Advanced feedback processing completed: user_id={user_id}, rating={rating}")
                        return {
                            "success": True,
                            "updated_profile": user_profile.copy(),
                            "style_adjustments": {"advanced_learning": True},
                            "feedback_processed": feedback_text,
                            "processing_method": "user_preferences_service"
                        }
            
            # Basic processing (ConversionService)
            conversion_service = self._get_service("conversion_service")
            if not conversion_service:
                return {
                    "success": False,
                    "error": "Feedback processing service is not initialized.",
                    "updated_profile": user_profile,
                    "processing_method": "none"
                }
            
            # Utilize ConversionService's basic feedback processing
            result = await conversion_service.process_user_feedback(
                feedback_text=feedback_text,
                user_profile=user_profile
            )
            result["processing_method"] = "conversion_service"
            return result
            
        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_profile": user_profile,
                "processing_method": "error"
            }
            
    def get_status(self) -> Dict:
        """Status information"""
        return {
            "rag_status": "ready" if self.is_initialized else "not_ready",
            "doc_count": self.vectorstore.index.ntotal if self.is_initialized else 0,
            "services_available": self.services_available
        }
