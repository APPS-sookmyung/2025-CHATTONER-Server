#!/usr/bin/env python3
"""
FastAPI Swagger UI 및 OpenAPI 설정
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any


def configure_swagger(app: FastAPI) -> None:
    """Swagger and OpenAPI configuration"""

    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        
        schema = get_openapi(
            title="ChatToner API",
            version="1.0.0",
            description="""
             **ChatToner** - AI-based Korean text personalization service
            
            ##  Key Features
            - **Text Style Conversion**: User-customized tone conversion (formal/friendly/neutral)
            - **Quality Analysis**: Automatic grammar, readability, and formality checking
            - **RAG System**: Document-based style guide search
            - **Personalization**: User feedback learning and profile adaptation
            
            ## 🛠 Usage Flow
            1. `/api/v1/conversion/convert` - Text conversion request
            2. `/api/v1/quality/analyze` - Quality score check  
            3. `/api/v1/feedback` - Feedback submission
            4. `/api/v1/rag/ask` - Style guide Q&A
            """,
            routes=app.routes,
        )

        schema["tags"] = [
            {"name": "health", "description": " Server status and connection check"},
            {"name": "conversion", "description": " AI-based text style conversion (core feature)"},
            {"name": "profile", "description": " User personalization profile management"},
            {"name": "quality", "description": " Text quality analysis (grammar/readability/formality)"},
            {"name": "feedback", "description": " User feedback collection and AI learning"},
            {"name": "rag", "description": " RAG-based document search and intelligent Q&A"},
        ]
        """
        # Security schema 
        schema["components"] = schema.get("components", {})
        schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }

        for path_data in schema["paths"].values():
            for operation in path_data.values():
                if "security" not in operation:
                    operation["security"] = [{"BearerAuth": []}]
        
        """
        
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi

def get_swagger_ui_parameters() -> Dict[str, Any]:
    """Swagger UI customization parameters"""
    return {
        "swagger_ui_parameters": {
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "operationsSorter": "method",
            "filter": True,
            "tagsSorter": "alpha",
            "tryItOutEnabled": True,
            "layout": "BaseLayout",
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "showExtensions": True,
            "showCommonExtensions": True,
        }
    }