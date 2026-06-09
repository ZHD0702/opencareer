class OpenCareerException(Exception):
    """OpenCareer 基础异常类"""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class SessionNotFoundException(OpenCareerException):
    """会话不存在异常"""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"会话 {session_id} 不存在",
            status_code=404
        )
        self.session_id = session_id


class MessageNotFoundException(OpenCareerException):
    """消息不存在异常"""
    
    def __init__(self, message_id: int):
        super().__init__(
            message=f"消息 {message_id} 不存在",
            status_code=404
        )
        self.message_id = message_id


class LLMServiceException(OpenCareerException):
    """LLM 服务异常"""
    
    def __init__(self, message: str = "LLM 服务暂时不可用"):
        super().__init__(message=message, status_code=503)
        self.message = message


class ValidationException(OpenCareerException):
    """数据验证异常"""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(message=message, status_code=422)
        self.field = field


class DatabaseException(OpenCareerException):
    """数据库操作异常"""
    
    def __init__(self, message: str = "数据库操作失败"):
        super().__init__(message=message, status_code=500)


class AuthenticationException(OpenCareerException):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败"):
        super().__init__(message=message, status_code=401)


class RateLimitException(OpenCareerException):
    """请求频率限制异常"""
    
    def __init__(self, message: str = "请求过于频繁，请稍后重试"):
        super().__init__(message=message, status_code=429)
