from sqlalchemy.exc import SQLAlchemyError
from requests.exceptions import Timeout

from rag_project.exceptions import DataBaseError, IngestionError, TimeOutError, UnexpectedError, RagError
from rag_project.logger import get_logger

logger = get_logger(__name__)


def db_session_manager(fn):
    def wrapper(self, *args, **kwargs):
        logger.info("session created")
        session = self.session_factory()
        error_log = None
        exc_info = True
        try:
            result = fn(self, session, *args, **kwargs)
            session.commit()
            return result

        except SQLAlchemyError as e:
            error_log = f"SQLAlchemy Error during transaction: {str(e)}"
            raise DataBaseError(message=error_log, code=500) from e

        except IngestionError as e:
            error_log = f"Ingestion Error during transaction: {str(e)}"
            exc_info = e.exc_info
            raise IngestionError(message=error_log, code=500) from e

        except RagError as e:
            error_log = f"Rag Error during transaction: {str(e)}"
            exc_info = e.exc_info
            raise IngestionError(message=error_log, code=500) from e

        except Timeout as e:
            error_log = f"TimeOut Error during transaction: {str(e)}"
            raise TimeOutError(message=error_log, code=500) from e

        except TypeError as e:
            error_log = f"TypeError Error during transaction: {str(e)}"
            raise DataBaseError(message=error_log, code=500) from e

        except Exception as e:
            error_log = f"Unexpected Error during transaction: {str(e)}"
            raise UnexpectedError(message=error_log, code=500) from e

        finally:
            if error_log:
                logger.info("session rollback")
                session.rollback()
                logger.error(error_log, exc_info=exc_info)

            self.reset_state()

            logger.info("session closed")
            session.close()

    return wrapper
