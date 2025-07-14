import os
import sys

sys.path.append(os.getcwd().rsplit('/', 1)[0])
from lpm_kernel.base.data import BaseData
from lpm_kernel.base.stage2_prompt import DESCRIPTION_SYSTEM_PROMPT, DESCRIPTION_USR
import os, json, random


class DescriptionData(BaseData):
    def __init__(self, processed_data_path: str = "resources/data/stage2/processed",
                 raw_data_path: str = "resources/data/stage2",
                 max_workers: int = 10, is_cot: bool = True):
        super().__init__(is_cot=is_cot, max_workers=max_workers)
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.context_limit = 30
        self.templates = {
            "CONCEPT": ["关于{entity_name}这个抽象概念，你有什么了解？", "你对{entity_name}这个抽象概念有什么了解吗？",
                        "你知道{entity_name}这个概念吗？", "你了解{entity_name}吗？",
                        "你知道关于{entity_name}这个概念的信息吗？"],
            "NORMAL_ENTITY": ["关于{entity_name}，你有什么了解？", "你对{entity_name}有什么了解吗？",
                              "你知道{entity_name}吗？", "你了解{entity_name}吗？", "你知道关于{entity_name}的信息吗？"],
            "PERSON": ["你对{entity_name}这个人有什么了解吗？", "你知道{entity_name}这个人吗？",
                       "你了解{entity_name}这个人吗？", "你知道关于{entity_name}这个人的信息吗？",
                       "你了解关于{entity_name}这个人的信息吗？"],
            "LOCATION": ["关于{entity_name}这个地理位置，你有什么了解？", "你对{entity_name}这个地理位置有什么了解吗？",
                         "你知道{entity_name}这个地理位置吗？", "你了解{entity_name}这个地理位置吗？",
                         "你知道关于{entity_name}这个地理位置的信息吗？",
                         "你了解关于{entity_name}这个地理位置的信息吗？"],
        }

        self.subjective_entities = self.load_entities()

    def load_entities(self):
        def get_note_content(note_id, notes):
            for note in notes:
                if note.get("noteId") == note_id:
                    return note.get("content", "")
            return ""

        entity_path = os.path.join(self.raw_data_path, "wiki", "wiki_res.json")
        with open(entity_path, 'r') as f:
            entities = json.load(f)
        notes_path = os.path.join(self.raw_data_path, "remade_note.json")
        with open(notes_path, 'r') as f:
            notes = json.load(f)

        for entity in entities:
            related_notes_ids = entity.get("related_notes", [])
            entity["notes"] = []
            for note_id in related_notes_ids:
                note_content = get_note_content(note_id, notes)
                if note_content:
                    entity["notes"].append(note_content)
        return entities

    def format_context(self, chunks):
        """
        Format the context information.
        """
        context = ""
        for idx, chunk in enumerate(chunks):
            context += f"Context: \n{chunk}\n\n"
        return context

    def build_messages(self):

        subjective_messages_list, subjective_questions = [], []
        for entity in self.subjective_entities:
            entity_name = entity["entityName"]
            entity_type = entity["entityType"]
            entity_description = entity["description"]
            template = random.choice(self.templates[entity_type])
            entity_context = self.format_context(entity["notes"][:self.context_limit])
            subjective_messages_list.append([{
                "role": "system",
                "content": DESCRIPTION_SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": DESCRIPTION_USR.format(question=template.format(entity_name=entity_name),
                                                  context=entity_context)
            }])
            subjective_questions.append(template.format(entity_name=entity_name))
        return subjective_messages_list, subjective_questions

    def build_qa_pairs(self, subjective_questions, subjective_responses):
        subjective_qa_pairs = []

        for subjective_question, subjective_response in zip(subjective_questions, subjective_responses):
            subjective_qa_pairs.append({"user": subjective_question, "assistant": subjective_response})
        return subjective_qa_pairs

    def save_qa_pairs(self, subjective_qa_pairs):
        os.makedirs(self.processed_data_path, exist_ok=True)
        subjective_qa_path = os.path.join(self.processed_data_path, "subjective_description.json")
        with open(subjective_qa_path, 'w') as f:
            json.dump(subjective_qa_pairs, f, ensure_ascii=False, indent=4)

    def run(self):

        subjective_messages_list, subjective_questions = self.build_messages()

        subjective_responses = self.build_responses(subjective_messages_list)
        # 3 build qa pairs
        subjective_qa_pairs = self.build_qa_pairs(
            subjective_questions, subjective_responses)
        # 4 save
        self.save_qa_pairs(subjective_qa_pairs)


if __name__ == "__main__":
    description = DescriptionData()
    description.run()
