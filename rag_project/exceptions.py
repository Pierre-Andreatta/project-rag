class IngestionError(Exception):
    def __init__(self, message: str, code: int = None, exc_info: bool = True):
        super().__init__(message)
        self.code = code
        self.exc_info = exc_info


class ScraperError(Exception):
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code


class DataBaseError(Exception):
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code


class TimeOutError(Exception):
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code


class UnexpectedError(Exception):
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code
