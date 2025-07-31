#!/usr/bin/env python3
"""
FastAPI Swagger UI 및 OpenAPI 설정
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any


def configure_swagger(app: FastAPI) -> None:
    """Swagger 및 OpenAPI 설정"""

    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        
        schema = get_openapi(
            title="Chat Toner API",
            version="1.0.0",
            description="Chat Toner는 AI 기반 한국어 텍스트 스타일 변환 서비스입니다.",
            routes=app.routes,
        )

        schema["tags"] = [
            {"name": "Health Check", "description": "서버 상태 및 연결 확인"},
            {"name": "Text Conversion", "description": "AI 기반 텍스트 스타일 변환"},
            {"name": "User Profile", "description": "사용자 개인화 프로필 관리"},
            {"name": "Preferences", "description": "네거티브 프롬프트 및 스타일 선호도"},
            {"name": "Advanced AI", "description": "파인튜닝 및 고급 프롬프트 기능"},
        ]
        """
        # 보안 스키마 
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
    """Swagger UI 커스터마이징 파라미터"""
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