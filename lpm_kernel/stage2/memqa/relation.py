import os
import sys

sys.path.append(os.getcwd().rsplit('/', 1)[0])
from lpm_kernel.base.data import BaseData
from lpm_kernel.base.stage2_prompt import RELATION_QUESTION_SYSTEM_PROMPT, RELATION_QUESTION_USR, \
    RELATION_ANSWER_SYSTEM_PROMPT, \
    RELATION_ANSWER_USR
import os, re, json
import argparse


class RelationshipData(BaseData):
    def __init__(self, raw_data_path: str = "resources/data/stage2",
                 processed_data_path: str = "resources/data/stage2/processed",
                 max_workers: int = 10, is_cot: bool = True,):
        super().__init__(is_cot=is_cot, max_workers=max_workers)
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.context_limit = 30
        self.subjective_notes = self.load_notes()
        self.subjective_relations = self.load_relations()

    def get_note_content(self, note_id, notes):
        note = notes.get(note_id, {})
        return note.get('content', note.get('insight', note.get('summary', '')))

    def format_context(self, related_notes, notes):
        """
        Format the context information from related notes.
        """
        context = ""
        for note_id in related_notes:
            note_content = self.get_note_content(note_id, notes)
            context += f"笔记 ID: {note_id}\n内容: {note_content}\n\n"
        return context

    def load_notes(self):

        subjective_note_path = os.path.join(self.raw_data_path, "remade_note.json")

        with open(subjective_note_path, 'r') as f:
            subjective_notes = json.load(f)

        subjective_notes = {note['noteId']: note for note in subjective_notes}
        return subjective_notes

    def load_relations(self):

        subjective_relation_path = os.path.join(self.raw_data_path, "wiki", "relations.json")

        with open(subjective_relation_path, 'r') as f:
            subjective_relations = json.load(f)
        return subjective_relations

    def build_question_messages(self):
        subjective_question_messages_list = []
        for relation in self.subjective_relations:
            source = relation["source"]
            target = relation["target"]
            relation_description = relation["description"]
            context = self.format_context(relation["related_notes"], self.subjective_notes)
            subjective_question_messages_list.append([{
                "role": "system",
                "content": RELATION_QUESTION_SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": RELATION_QUESTION_USR.format(source=source, target=target,
                                                        relation_description=relation_description, context=context)
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
        for i, relation in enumerate(self.subjective_relations):
            source = relation["source"]
            target = relation["target"]
            relation_description = relation["description"]
            context = self.format_context(relation["related_notes"], self.subjective_notes)[:self.context_limit]
            subjective_answer_messages_list.append([{
                "role": "system",
                "content": RELATION_ANSWER_SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": RELATION_ANSWER_USR.format(source=source, target=target,
                                                      relation_description=relation_description, context=context,
                                                      question=subjective_questions[i])
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
            subjective_qa_pairs.append({"user": subjective_question, "assistant": subjective_answer})
        return subjective_qa_pairs

    def save_qa_pairs(self, subjective_qa_pairs):
        os.makedirs(self.processed_data_path, exist_ok=True)

        subjective_qa_pairs_path = os.path.join(self.processed_data_path, "subjective_relation.json")

        with open(subjective_qa_pairs_path, "w") as f:
            json.dump(subjective_qa_pairs, f, ensure_ascii=False, indent=4)

    def run(self):

        subjective_relations = self.load_relations()

        subjective_question_messages_list = self.build_question_messages()
        subjective_questions = self.build_question_response(
            subjective_question_messages_list)
        # 3 build answer messages and responses
        subjective_answer_messages_list = self.build_answer_messages(
            subjective_questions)
        subjective_answers = self.build_answer_response(
            subjective_answer_messages_list)
        # 4 build question and answer pairs
        subjective_qa_pairs = self.build_qa_pairs(
            subjective_questions, subjective_answers)
        # 5 save question and answer pairs
        self.save_qa_pairs(subjective_qa_pairs)


if __name__ == "__main__":
    relation = RelationshipData()
    relation.run()
