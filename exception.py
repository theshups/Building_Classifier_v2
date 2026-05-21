"""
exception.py — Custom exception with file + line number tracking.
"""
from logger import get_logger

log = get_logger(__name__)


class AppException(Exception):
    def __init__(self, error, error_sys=None):
        super().__init__(str(error))
        if error_sys is not None:
            _, _, tb = error_sys.exc_info()
            self.error_message = (
                f"[{tb.tb_frame.f_code.co_filename}] "
                f"line {tb.tb_lineno}: {error}"
            ) if tb else str(error)
        else:
            self.error_message = str(error)
        log.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
