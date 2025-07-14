import os
import sys

sys.path.append(os.getcwd().rsplit('/', 1)[0])
from lpm_kernel.base.data import BaseData
from lpm_kernel.base.stage2_prompt import ENTITY_QUESTION_SYSTEM_PROMPT, ENTITY_QUESTION_USR, \
    ENTITY_ANSWER_SYSTEM_PROMPT, \
    ENTITY_ANSWER_USR
import os, re, json


class EntityData(BaseData):
    def __init__(self, raw_data_path: str = "resources/data/stage2",
                 processed_data_path: str = "resources/data/stage2/processed",
                 max_workers: int = 10, is_cot: bool = True,):
        super().__init__(is_cot=is_cot,max_workers=max_workers)
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.context_limit = 30
        self.subjective_entities = self.load_entities()

    def get_note_content(self, note_id, notes):
        """
        Get the content of a note by its ID.
        """
        for note in notes:
            if note.get("noteId") == note_id:
                return note.get("content", "")
        return ""

    def format_context(self, chunks):
        """
        Format the context information.
        """
        context = ""
        for idx, chunk in enumerate(chunks):
            context += f"Context: \n{chunk}\n\n"
        return context

    def load_entities(self):

        subjective_note_path = os.path.join(self.raw_data_path, "remade_note.json")

        subjective_entity_path = os.path.join(self.raw_data_path, "wiki", "wiki_res.json")

        with open(subjective_note_path, 'r') as f:
            subjective_notes = json.load(f)

        with open(subjective_entity_path, 'r') as f:
            subjective_entities = json.load(f)

        for subjective_entity in subjective_entities:

            related_notes_ids = subjective_entity.get("related_notes", [])
            subjective_entity["notes"] = []
            for note_id in related_notes_ids:
                note_content = self.get_note_content(note_id, subjective_notes)
                if note_content:
                    subjective_entity["notes"].append(note_content)
        return subjective_entities

    def build_question_messages(self):

        subjective_question_messages_list = []
        for entity in self.subjective_entities:
            entity_name = entity["entityName"]
            entity_type = entity["entityType"]
            entity_description = entity["description"]
            entity_context = self.format_context(entity["notes"][:self.context_limit])
            subjective_question_messages_list.append([{
                "role": "system",
                "content": ENTITY_QUESTION_SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": ENTITY_QUESTION_USR.format(entity_name=entity_name, entity_type=entity_type,
                                                      entity_description=entity_description,
                                                      entity_context=entity_context)
            }])
        return subjective_question_messages_list

    def build_question_response(self, subjective_question_messages_list):

        subjective_question_responses = self.build_responses(subjective_question_messages_list)
        subjective_questions = []

        for subjective_question_response in subjective_question_responses:
            if not isinstance(subjective_question_response, (str, bytes)):
                subjective_question_response = ""
            subjective_question_match = re.search(r"<question>([^<]*?)</question>", subjective_question_response,
                                                  re.DOTALL)
            if subjective_question_match:
                subjective_questions.append(subjective_question_match.group(1).strip())
            else:
                subjective_questions.append("")
        return subjective_questions

    def build_answer_messages(self, subjective_questions):

        subjective_answer_messages_list = []
        for i, entity in enumerate(self.subjective_entities):
            entity_name = entity["entityName"]
            entity_type = entity["entityType"]
            entity_description = entity["description"]
            entity_context = self.format_context(entity["notes"][:self.context_limit])
            subjective_answer_messages_list.append([{
                "role": "system",
                "content": ENTITY_ANSWER_SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": ENTITY_ANSWER_USR.format(entity_name=entity_name, entity_type=entity_type,
                                                    entity_description=entity_description,
                                                    entity_context=entity_context, question=subjective_questions[i])
            }])
        return subjective_answer_messages_list

    def build_answer_response(self, subjective_answer_messages_list):

        subjective_answer_responses = self.build_responses(subjective_answer_messages_list)
        subjective_answers = []

        for subjective_answer_response in subjective_answer_responses:
            if not isinstance(subjective_answer_response, (str, bytes)):
                subjective_answer_response = ""
            subjective_answers.append(subjective_answer_response.strip())
        return subjective_answers

    def build_qa_pairs(self, subjective_questions, subjective_answers):

        subjective_qa_pairs = []

        for subjective_question, subjective_answer in zip(subjective_questions, subjective_answers):
            if subjective_answers == "":
                continue
            subjective_qa_pairs.append({"user": subjective_question, "assistant": subjective_answer})
        return subjective_qa_pairs

    def save_qa_pairs(self, subjective_qa_pairs):
        os.makedirs(self.processed_data_path, exist_ok=True)
        subjective_qa_pairs_path = os.path.join(self.processed_data_path, "subjective_entity.json")

        with open(subjective_qa_pairs_path, "w") as f:
            json.dump(subjective_qa_pairs, f, ensure_ascii=False, indent=4)

    def run(self):

        subjective_question_messages_list = self.build_question_messages()
        subjective_questions = self.build_question_response(
            subjective_question_messages_list)

        subjective_answer_messages_list = self.build_answer_messages(
            subjective_questions)
        subjective_answers = self.build_answer_response(
            subjective_answer_messages_list)

        subjective_qa_pairs = self.build_qa_pairs(
            subjective_questions, subjective_answers)

        self.save_qa_pairs(subjective_qa_pairs)


if __name__ == "__main__":
    entity = EntityData()
    entity.run()
