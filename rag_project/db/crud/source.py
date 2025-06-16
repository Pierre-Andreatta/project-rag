from typing import Optional
from sqlalchemy import select

from rag_project.db.models.source import SourceORM, RejectReasonORM
from rag_project.db.crud.base_crud import BaseCRUD
from rag_project.domain.models import SourceTypeEnum
from rag_project.logger import get_logger


logger = get_logger(__name__)


class SourceCRUD(BaseCRUD):

    def create_source(self, source_path: str, source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT) -> SourceORM:
        source = SourceORM(source_path=source_path, source_type=source_type)
        self.session.add(source)
        return source

    def get_or_create_source(self, source_path: str, source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT) -> SourceORM:
        source = self.get_source_by_path_to_content(path=source_path)

        if not source:
            source = self.create_source(
                source_path=source_path,
                source_type=source_type
            )
            self.session.flush()

        return source

    def get_source_by_path_to_content(self, path: str) -> Optional[SourceORM]:
        stmt = select(SourceORM).where(path == SourceORM.source_path)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_source_by_id(self, source_id: int) -> Optional[SourceORM]:
        return self.session.get(SourceORM, source_id)

    def approve_source(self, source_id: int):
        self.session.query(
            SourceORM
        ).filter(
            SourceORM.id == source_id
        ).update(
            {'is_accepted': True}
        )

    def reject_source(self, source_id: int, reason: int):
        stmt = select(RejectReasonORM).where(RejectReasonORM.reason == reason)
        reason_obj = self.session.execute(stmt).scalar_one_or_none()

        if not reason_obj:
            raise ValueError(f"Invalid reason: {reason}")

        self.session.query(
            SourceORM
        ).filter(
            SourceORM.id == source_id
        ).update(
            {'is_accepted': False, 'rejection_reason_id': reason_obj.id}
        )

    def list_sources(self, *,
                     only_accepted: bool = None,
                     source_type: SourceTypeEnum = None,
                     limit: int = 100) -> list[SourceORM]:

        query = select(SourceORM)

        if only_accepted is not None:
            query = query.where(SourceORM.is_accepted == only_accepted)

        if source_type is not None:
            query = query.where(SourceORM.source_type == source_type)

        query = query.limit(limit)

        return list(self.session.execute(query).scalars())
