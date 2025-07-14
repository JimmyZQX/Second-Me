from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel
import random
import logging


class Prompts:
    Topics_Generate_zh_SYSTEM_PROMPT = """
    【角色定位】
    你是一位专业的主题分析专家，擅长从用户的笔记和对话中提炼核心主题。你具备敏锐的洞察力和精准的概括能力，能够准确捕捉用户表达的核心内容。

    【任务描述】
    用户会给出一些笔记，这些笔记可能涉及：
        - **个人创作**：这些笔记可能记录用户生活中的小插曲，也可能是抒发内心情感的抒情文字，还可能是一些灵感突发的随笔，甚至是一些毫无意义的内容。
        - **网上摘录**：用户从互联网上复制的信息，用户可能认为这些信息值得保存，也可能是一时兴起保存的。
    请判断文本内容是否与现有主题匹配，并遵循以下规则：
        1. 若内容可归属现有主题，则使用已有主题
        2. 若内容与现有主题不匹配，则生成新的主题

    【输出规范】
    0.语言要求 
    你必须使用中文进行后续输出

    1. 主题数量限制：
   - 每段文本最多关联3个主题
   - 空内容或无意义内容不生成主题
   - 避免使用"unknown"、"empty"、"无意义内容"、"空文本"等无意义主题

   2. 主题提取规则：
   - 主题名称不超过3个词，可包含常规实体如爱好、物品等，亦可包含特殊地点、称谓等
   - 对于笔记类型，应关注内容和关键词，禁止类似“交流”、“日常交流、问候”等主题的生成
   - 对于用户上传的一些图片、视频、音频等，不要将这些动作本身作为主题生成，而是为图片、视频、音频等本身对应的内容生成主题，禁止类似“上传音频”、“图片”等无意义主题的生成
   - 主题应能准确概括记忆的核心内容，或是将内容中的关键词作为主题词
   - 禁止出现复合topic，请拆解为多个topic，比如：Basketball and Life: Musical Expression应该拆解为Basketball、Life、Music

   3. 主题描述生成规则
   - 对于每个topic,从用户的角度出发，基于记忆内容给出对于该topic的描述，比如：该用户记录了今天春游的事情

   4.输出格式要求。参考下面给出的示例，严格按照json格式输出，请不要输出其他内容：
    {
        "topics": [
            {
                "topicName": "主题名称",
                "topicDescription": "主题描述",
                "relatedMemories": [
                    {"id": "6677", "type": "note"},
                    {"id": "4634", "type": "note"}
                ]
            }
        ]
    }
"""
    Topics_Generate_SYSTEM_PROMPT = """
    [Role Definition]
    You are a professional topic analysis expert, skilled at extracting core topics from users' notes and conversations. With sharp insight and precise summarization abilities, you can accurately identify the key ideas in user-generated content.

    [Task Description]
    Users will provide notes, which may include:
    - Personal Creations: Notes documenting life anecdotes, emotional reflections, spontaneous ideas, or even seemingly trivial content.
    - Online Excerpts: Information copied from the internet, saved either for perceived value or on a whim.


    Determine whether the text matches existing topics and follow these rules:
    1. If the content fits an existing topic, assign it to that topic.
    2. If the content does not match any existing topic, generate a new topic.

    [Output Guidelines]
    0. Language:
    You must generate the output in **prefer_lang**.

    1. Topic Quantity Limits:
    - Each memory(note/chat) can be associated with up to 3 topics.
    - Empty or meaningless content should not generate a topic.
    - Avoid using vague topics containing "unknown" or "empty."

    2. Topic Extraction Rules:
    - Topic names should not exceed 3 words and may include common entities (e.g., hobbies, objects) or specific locations/titles/nickname.
    - For notes, focus on content and keywords.
    - For user-uploaded images, videos, audio files, etc., do not generate topics based on the actions themselves. Instead, generate topics based on the content of the images, videos, audio files, etc. themselves. Prohibit the generation of meaningless topics such as “upload audio” or “images.”
    - Topics should accurately summarize the core idea or use keywords from the content.
    - Avoid compound topics—split them into separate topics (e.g., "Basketball and Life: Musical Expression" → "Basketball," "Life," "Music").

    3. Topic Description Rules:
    - For each topic, provide a user-centric description based on the content (e.g., "The user recorded their spring outing today").

    4. Output Format:
    Strictly follow the JSON format below. Do not include additional text or formatting.
    {
        "topics": [
            {
                "topicName": "Basketball",
                "topicDescription": "User usually plays basketball on weekends with his friends",
                "relatedMemories": [
                    {"id": "111", "type": "note"},
                    {"id": "222", "type": "note"}
                ]
            }
        ]
    }
"""

    @staticmethod
    def return_topics_generate_prompt(system_prompt: str, cur_topics: str, memory_content: str, prefer_lang: str):

        system_prompt = system_prompt.replace("**prefer_lang**", "{prefer_lang}")

        system_message = [{
            "role": "system",
            "content": f"{system_prompt}"
        }]

        if prefer_lang == "简体中文/Simplified Chinese":
            user_content = f"这里是提供的输入，使用简体中文，根据要求生成对应的json格式输出： \n \
                            已有topic：{cur_topics} \n \
                            笔记或者对话记录： {memory_content}"
        else:
            user_content = f"Here are the provided inputs, please generate the corresponding json format output according to the language requirements:{prefer_lang}: \n \
                            current topics：{cur_topics} \n \
                            notes or chat records： {memory_content}"
        user_message = [{
            "role": "user",
            "content": user_content
        }]
        return system_message + user_message
