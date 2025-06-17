from enum import Enum


class SourceTypeEnum(str, Enum):
    # HACK: Domain class used in Data layer
    DEFAULT = 'DEFAULT'
    WEB = "web"
    PDF = "pdf"
    YOUTUBE = "youtube"


class RejectReasonEnum(str, Enum):
    # HACK: Domain class used in Data layer
    INAPPROPRIATE = "inappropriate"
    DUPLICATE = "duplicated"
    LOW_QUALITY = "low_quality"
    OUTDATED = "obsolete"


class LanguageEnum(str, Enum):
    FR = "fr"
    EN = "en"
