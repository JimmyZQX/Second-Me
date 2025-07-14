import json
import math
import uuid
import logging
import datetime
import numpy as np

from enum import Enum
from typing import List, Optional, Tuple, Union, Any, Dict


class ConfidenceLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


CONFIDENCE_LEVELS_INT = {
    ConfidenceLevel.VERY_LOW: 1,
    ConfidenceLevel.LOW: 2,
    ConfidenceLevel.MEDIUM: 3,
    ConfidenceLevel.HIGH: 4,
    ConfidenceLevel.VERY_HIGH: 5
}

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class ShadeTimeline:
    def __init__(self,
                 refMemoryId: int = None,
                 createTime: str = "",
                 descSecondView: str = "",
                 descThirdView: str = "",
                 is_new: bool = False):
        self.create_time = createTime
        self.ref_memory_id = refMemoryId
        self.desc_second_view = descSecondView
        self.desc_third_view = descThirdView
        self.is_new = is_new

    @classmethod
    def from_raw_format(cls, raw_format: Dict[str, Any]):
        return cls(
            refMemoryId=raw_format.get("refMemoryId", None),
            createTime=raw_format.get("createTime", ""),
            descSecondView="",
            descThirdView=raw_format.get("description", ""),
            is_new=True
        )

    def add_second_view(self, description):
        self.desc_second_view = description

    def to_json(self):
        return {
            "createTime": self.create_time,
            "refMemoryId": self.ref_memory_id,
            "descThirdView": self.desc_third_view,
            "descSecondView": self.desc_second_view
        }


class ShadeInfo:
    def __init__(self,
                 id: int = None,
                 shadeName: str = "",
                 icon: str = "",
                 shadeDescription: str = "",
                 shadeDescriptionThirdView: str = "",
                 shadeContent: str = "",
                 shadeContentThirdView: str = "",
                 timelines: List[Dict[str, Any]] = [],
                 confidenceLevel: str = "",
                 **kwargs):
        self.id = id
        self.name = shadeName
        self.icon = icon
        self.desc_second_view = shadeDescription
        self.desc_third_view = shadeDescriptionThirdView
        self.content_third_view = shadeContent
        self.content_second_view = shadeContentThirdView
        if confidenceLevel:
            self.confidence_level = ConfidenceLevel(confidenceLevel)
        else:
            self.confidence_level = None

        self.timelines = [ShadeTimeline(**timeline) for timeline in timelines]

    def imporve_shade_info(self, improveDesc: str, improveContent: str, improveTimelines: List[Dict[str, Any]]):
        self.desc_third_view = improveDesc
        self.content_third_view = improveContent
        self.timelines.extend([ShadeTimeline.from_raw_format(timeline) for timeline in improveTimelines])

    def add_second_view(self, domainDesc: str, domainContent: str, domainTimeline: List[Dict[str, Any]]):
        self.desc_second_view = domainDesc
        self.content_second_view = domainContent
        timelime_dict = {timelime.ref_memory_id: timelime for timelime in self.timelines}
        for timeline in domainTimeline:
            ref_memory_id = timeline.get("refMemoryId", None)
            if not (ref_memory_id and ref_memory_id in timelime_dict):
                logging.error(f"Timeline with refMemoryId {ref_memory_id} already exists, skipping")
                continue
            timelime_dict[ref_memory_id].add_second_view(timeline.get("description", ""))

    def _preview_(self, second_view: bool = False):
        if second_view:
            return f"- **{self.name}**: {self.desc_second_view}"
        return f"- **{self.name}**: {self.desc_third_view}"

    def to_str(self):
        shade_statement = f"---\n**[Name]**: {self.name}\n**[Icon]**: {self.icon}\n"
        shade_statement += f"**[Description]**: \n{self.desc_third_view}\n\n**[Content]**: \n{self.content_third_view}\n"
        shade_statement += "---\n\n[Timelines]:\n"
        for timeline in self.timelines:
            shade_statement += f"- {timeline.create_time}, {timeline.desc_third_view}, {timeline.ref_memory_id}\n"
        return shade_statement

    def to_json(self):
        return {
            "id": self.id,
            "shadeName": self.name,
            "icon": self.icon,
            "shadeDescription": self.desc_second_view,
            "shadeDescriptionThirdView": self.desc_third_view,
            "shadeContent": self.content_third_view,
            "shadeContentThirdView": self.content_second_view,
            "confidenceLevel": self.confidence_level if self.confidence_level else None,
            "timelines": [timeline.to_json() for timeline in self.timelines]
        }


class AttributeInfo:
    def __init__(self,
                 id: int = None,
                 name: str = "",
                 description: str = "",
                 confidenceLevel: Optional[Union[str, ConfidenceLevel]] = None):
        self.id = id
        self.name = name
        self.description = description
        if confidenceLevel and isinstance(confidenceLevel, str):
            self.confidence_level = ConfidenceLevel(confidenceLevel)
        elif isinstance(confidenceLevel, ConfidenceLevel):
            self.confidence_level = confidenceLevel
        else:
            self.confidence_level = None

    def to_str(self):
        # - **[Attribute Name]**: (Attribute Description), Confidence level: [LOW/MEDIUM/HIGH]
        return f"- **{self.name}**: {self.description}, Confidence level: {self.confidence_level.value}"

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "confidenceLevel": self.confidence_level.value if self.confidence_level else None
        }


class GlobalBioResponse:
    def __init__(self, result: Any, success: bool):
        self.success: bool = success
        self.message: str = ""
        self.global_bio: Optional[Dict[str, Any]] = None
        self.cluster_list: Optional[List[Dict[str, Any]]] = None

        if not success:
            self.message = result if isinstance(result, str) else "Error occurred"
            logging.error(self.message)
        else:
            self.message = "Success"
            self.global_bio = result.get("globalBio")
            self.cluster_list = result.get("clusterList")

    def to_json(self) -> str:
        return {
            "success": self.success,
            "message": self.message,
            "globalBio": self.global_bio,
            "clusterList": self.cluster_list
        }


class Bio:
    def __init__(self,
                 contentThirdView: str = "",
                 content: str = "",
                 summaryThirdView: str = "",
                 summary: str = "",
                 attributeList: List[Dict[str, Any]] = [],
                 shadesList: List[Dict[str, Any]] = []):
        self.content_third_view = contentThirdView
        self.content_second_view = content
        self.summary_third_view = summaryThirdView
        self.summary_second_view = summary
        self.attribute_list = sorted([AttributeInfo(**attribute) for attribute in attributeList],
                                     key=lambda x: CONFIDENCE_LEVELS_INT[x.confidence_level], reverse=True)
        self.shades_list = sorted([ShadeInfo(**shade) for shade in shadesList], key=lambda x: len(x.timelines),
                                  reverse=True)

    def to_str(self) -> str:
        global_bio_statement = ""
        if self.is_raw_bio():
            global_bio_statement += f"**[Origin Analysis]**\n{self.summary_third_view}\n"
        # global_bio_statement += f"**[Identity Attributes]**\n"
        # global_bio_statement += '\n'.join([attribute.to_str() for attribute in self.attribute_list])

        global_bio_statement += f"\n**[Current Shades]**\n"
        for shade in self.shades_list:
            global_bio_statement += shade.to_str()
            global_bio_statement += "\n==============\n"
        return global_bio_statement

    def complete_content(self, second_view: bool = False) -> str:
        interests_preference_field = "\n### User's Interests and Preferences ###\n" + '\n'.join(
            [shade._preview_(second_view) for shade in self.shades_list])
        if not second_view:
            conclusion_field = "\n### Conclusion ###\n" + self.summary_third_view
        else:
            conclusion_field = "\n### Conclusion ###\n" + self.summary_second_view
        return f"""## Comprehensive Analysis Report ##
{interests_preference_field}
{conclusion_field}"""

    def is_raw_bio(self) -> bool:
        if not self.content_third_view and not self.summary_third_view:
            return True
        return False

    def to_json(self) -> Dict[str, Any]:
        return {
            "contentThirdView": self.content_third_view,
            "content": self.content_second_view,
            "summaryThirdView": self.summary_third_view,
            "summary": self.summary_second_view,
            "shadesList": [shade.to_json() for shade in self.shades_list]
        }


def gen_uuid() -> str:
    return str(uuid.uuid4())


def get_now() -> str:
    return datetime.datetime.now().strftime(TIME_FORMAT)