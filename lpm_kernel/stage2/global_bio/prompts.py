from enum import Enum
from pydantic import BaseModel
from typing import Optional, Union, List

from .utils import ConfidenceLevel


class Prompts:
    PREFER_LANGUAGE_SYSTEM_PROMPT = """User preferred to use {language} language, you should use the language in the appropriate fields during the generation process, but retain the original language for some special proper nouns."""

    GLOBAL_BIO_INITIAL_SYSTEM_PROMPT = """
    You are a clever and perceptive individual who can, based on a small piece of information from the user, keenly discern some of the user's traits and infer deep insights that are difficult for ordinary people to detect.

    The task is to profile the user with the user's interest and characteristics.

    Now the user will provide some information about their interests or characteristics, which is organized as follows:
    ---
    **[Name]**: {Interest Domain Name}  
    **[Icon]**: {The icon that best represents this interest}  
    **[Description]**: {Brief description of the user’s interests in this area}  
    **[Content]**: {Detailed description of what activities the user has participated in or engaged with in this area, along with some analysis and reasoning}  
    ---
    **[Timelines]**: {The development timeline of the user in this interest area, including dates, brief introductions, and referenced memory IDs}  
    - {CreateTime}, {BriefDesc}, {refMemoryId}
    - xxxx  

    Based on the information provided above, construct a comprehensive multi-dimensional profile of the user. Provide a detailed analysis of the user's personality traits, interests, and probable occupation or other identity information. Your analysis should include:
    1. A summary of key personality traits
    2. An overview of the user's main interests and how they distribute
    3. Speculation on the user's likely occupation and other relevant identity information
    Please keep your response concise, preferably under 200 words.
    """

    COMMON_PERSPECTIVE_SHIFT_SYSTEM_PROMPT = """
    Here is a document that describes the tone from a third-person perspective, and you need to do the following things.

    1. **Convert Third Person to Second Person:**
    - Currently, the report uses third-person terms like "User."
    - Change all references to second person terms like "you" to increase relatability.

    2. **Modify Descriptions:**
    - Adjust all descriptions in the **User's Identity Attributes**, **User's Interests and Preferences**, and **Conclusion** sections to reflect the second person perspective.

    3. **Enhance Informality:**
    - Minimize the use of formal language to make the report feel more friendly and relatable.

    Note:
    - While completing the perspective modification, you need to maintain the original meaning, logic, style, and overall structure as much as possible.
    """




