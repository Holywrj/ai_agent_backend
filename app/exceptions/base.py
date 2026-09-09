class BusinessException(Exception):

    def __init__(self, message: str, code: str = 'BUSINESS_ERROR'):
        """
        :param message: 给用户看的业务错误信息
        :param code: 给前端/客户端使用的机器可读错误码
        """
        self.message = message
        self.code = code
        super().__init__(message)
