from typing import Optional
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from rag_project.db.models.source import SourceORM, RejectReasonORM
from rag_project.db.crud.base_crud import BaseCRUD
from rag_project.dto.models import SourceTypeEnum, SourceDto
from rag_project.exceptions import ValidationError, DataBaseError
from rag_project.logger import get_logger


logger = get_logger(__name__)


class SourceCRUD(BaseCRUD):

    def create_source(self, source_path: str, source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT) -> SourceDto:
        source_orm = SourceORM(source_path=source_path, source_type=source_type)
        self.session.add(source_orm)
        self.session.flush()
        return SourceDto.from_orm(source_orm)

    def get_or_create_source(self, source_path: str, source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT) -> SourceDto:

        if not source_path or not source_path.strip():
            raise ValidationError("Source path cannot be empty", field="source_path")

        if not isinstance(source_type, SourceTypeEnum):
            raise ValidationError(f"Invalid source type: {source_type}", field="source_type")

        try:
            # TODO: handle source update if content is added to existing source
            #       -> sources.updated_at
            #       Currently : sources not updated and content added in addition to the previous one
            source = self.get_source_by_path_to_content(path=source_path)

            if not source:
                logger.info(f"Source not found, creating new one: {source_path}")
                source = self.create_source(
                    source_path=source_path,
                    source_type=source_type
                )
            else:
                logger.debug(f"Found existing source: {source_path}")

            return source

        except ValidationError:
            raise
        except DataBaseError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_or_create_source for {source_path}: {e}", exc_info=True)
            raise DataBaseError(f"Failed to get or create source: {str(e)}") from e

    def get_source_by_path_to_content(self, path: str) -> Optional[SourceDto]:

        if not path or not path.strip():
            raise ValidationError("Path cannot be empty", field="path")

        try:
            stmt = select(SourceORM).where(SourceORM.source_path == path.strip())
            source_orm = self.session.execute(stmt).scalar_one_or_none()

            if source_orm:
                return SourceDto.from_orm(source_orm)

            logger.debug(f"No source found for path: {path}")

        except SQLAlchemyError as e:
            message = f"Failed to get source by path {path}: {str(e)}"
            logger.error(message)
            raise DataBaseError(message) from e

    def get_source_by_id(self, source_id: int) -> Optional[SourceDto]:

        if not isinstance(source_id, int) or source_id <= 0:
            raise ValidationError(f"Invalid source ID: {source_id}", field="source_id")

        try:
            source_orm = self.session.get(SourceORM, source_id)
            if source_orm:
                return SourceDto.from_orm(source_orm)
            logger.warning(f"get_source_by_id: {source_id} not found")
        except SQLAlchemyError as e:
            message = f"Failed to get source by ID {source_id}: {e}"
            logger.error(message)
            raise DataBaseError(message) from e

    def approve_source(self, source_id: int):

        if not isinstance(source_id, int) or source_id <= 0:
            raise ValidationError(f"Invalid source ID: {source_id}", field="source_id")

        try:

            source = self.get_source_by_id(source_id)
            if not source:
                message = f"Source: {source_id} not found"
                logger.error(message)
                raise DataBaseError(message)

            rows_updated = self.session.query(
                SourceORM
            ).filter(
                SourceORM.id == source_id
            ).update({
                'is_accepted': True,
                'rejection_reason_id': None
            })

            if rows_updated > 0:
                logger.info(f"Approved source: {source_id}")

        except SQLAlchemyError as e:
            message = f"Failed to approve source {source_id}: {e}"
            logger.error(message)
            raise DataBaseError(message) from e

    def reject_source(self, source_id: int, reason: int):

        if not isinstance(source_id, int) or source_id <= 0:
            raise ValidationError(f"Invalid source ID: {source_id}", field="source_id")

        if not isinstance(reason, int) or reason <= 0:
            raise ValidationError(f"Invalid reason ID: {reason}", field="reason")

        try:

            source = self.get_source_by_id(source_id)
            if not source:
                logger.warning(f"Cannot reject non-existent source: {source_id}")
                # TODO: raise error ?

            stmt = select(RejectReasonORM).where(RejectReasonORM.reason == reason)  # TODO: by id ?
            reason_obj = self.session.execute(stmt).scalar_one_or_none()

            if not reason_obj:
                raise ValidationError(f"Invalid reason: {reason}", field="reason")

            rows_updated = self.session.query(
                SourceORM
            ).filter(
                SourceORM.id == source_id
            ).update({
                'is_accepted': False,
                'rejection_reason_id': reason_obj.id
            })

            if rows_updated > 0:
                logger.info(f"Rejected  source: {source_id}")

        except SQLAlchemyError as e:
            message = f"Failed to approve source {source_id}: {e}"
            logger.error(message)
            raise DataBaseError(message) from e

    def list_sources(self, *,
                     only_accepted: bool = None,
                     source_type: SourceTypeEnum = None,
                     limit: int = 100) -> list[SourceDto]:

        # TODO: to rework

        query = select(SourceORM)

        if only_accepted is not None:
            query = query.where(SourceORM.is_accepted == only_accepted)

        if source_type is not None:
            query = query.where(SourceORM.source_type == source_type)

        query = query.limit(limit)
        source_orm = list(self.session.execute(query).scalars())
        return SourceDto.from_orm(source_orm)
