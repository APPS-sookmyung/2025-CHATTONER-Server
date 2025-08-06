"""
기본 서비스 클래스
모든 서비스의 공통 기능을 제공하는 베이스 클래스
"""

import logging
from abc import ABC


class BaseService(ABC):
    """모든 서비스의 기본 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
    
    def _log_error(self, message: str, exception: Exception = None):
        """에러 로깅 헬퍼 메서드"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}")
        else:
            self.logger.error(message)
    
    def _log_info(self, message: str):
        """정보 로깅 헬퍼 메서드"""
        self.logger.info(message)
    
    def _log_debug(self, message: str):
        """디버그 로깅 헬퍼 메서드"""
        self.logger.debug(message)