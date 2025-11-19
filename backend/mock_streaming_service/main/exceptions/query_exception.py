class QueryException(Exception):
    def __init__(self, message, invalid_value):
        super().__init__(message)
        self.invalid_value = invalid_value