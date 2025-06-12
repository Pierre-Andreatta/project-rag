class IngestionError(Exception):
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code


class ScraperError(Exception):
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code


class DataBaseError(Exception):
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code
