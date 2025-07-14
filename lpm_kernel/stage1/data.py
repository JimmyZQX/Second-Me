import json
import os
import random
import re

from lpm_kernel.base.data import BaseData
from lpm_kernel.base.stage1_prompt import subjective_retell_prompt, subjective_insight_prompt
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.file_data.document_repository import DocumentRepository

logger = get_train_process_logger()

class Stage1Data(BaseData):
    def __init__(self, raw_data_path: str = "lpm_kernel/base/all_general.json",
                 processed_data_path: str = "resources/data/stage1", max_workers: int = 10, is_cot: bool = True):
        super().__init__(is_cot=is_cot, max_workers=max_workers)
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def build_messages(self):
        """
        1 Get subjective data
        2 Rephrase question data construction
        3 Insight question data construction
        """
        subjective_messages_list = []

        try:
            doc_repository = DocumentRepository()
            documents = doc_repository.list()

            logger.info(f"Found {len(documents)} documents for message building")

            for doc in documents:
                if doc.raw_content:
                    memory_content = f"这是一段你过去的记忆：{doc.raw_content}"
                    subjective_messages_list.append([{"role": "system", "content": subjective_retell_prompt},
                                                     {"role": "user", "content": memory_content}])
                    subjective_messages_list.append([{"role": "system", "content": subjective_insight_prompt},
                                                     {"role": "user", "content": memory_content}])

            logger.info(f"Successfully processed {len(subjective_messages_list)} documents with content")

            return subjective_messages_list

        except Exception as e:
            logger.error(f"Error building messages from documents: {str(e)}", exc_info=True)
            return []

    def parse_subjective_data(self, messages_list, responses):
        results = []
        for messages, response in zip(messages_list, responses):

            if not isinstance(response, str):
                response = ""

            question_match = re.search(r"<question>([^<]*?)</question>", response, re.DOTALL)
            response_match = re.search(r"<response>([^<]*?)</response>", response, re.DOTALL)
            summary_match = re.search(r"<summary>([^<]*?)</summary>", response, re.DOTALL)

            if question_match:
                question = question_match.group(1)
                if summary_match:
                    response = "<think>好的，我曾经说过：" + messages[1][
                        "content"] + "</think>\n <answer>" + summary_match.group(1) + "</answer>"
                elif response_match:
                    response = "<think>好的，我曾经说过：" + messages[1][
                        "content"] + "</think>\n <answer>" + response_match.group(1) + "</answer>"
            else:
                question, response = "", ""
            info = {"system": "", "user": question, "assistant": response}
            results.append(info)
        return results

    def load_final_general_data(self, sampled_num: int = 2):
        results = []
        general_data_path = os.path.join(self.raw_data_path)
        with open(general_data_path, "r") as f:
            shuffled_data = json.load(f)
            random.shuffle(shuffled_data)
            sampled_data_list = shuffled_data[:sampled_num]

        for sampled_data in sampled_data_list:
            info = {"system": sampled_data["system"], "user": sampled_data["user"],
                    "assistant": sampled_data["assistant"]}
            results.append(info)

        return results

    def save_data(self, subjective_data, general_data):
        epoch1_date = subjective_data+general_data
        random.shuffle(epoch1_date)
        os.makedirs(self.processed_data_path, exist_ok=True)
        subjective_data_path = os.path.join(self.processed_data_path, "subjective.json")
        general_data_path = os.path.join(self.processed_data_path, "general.json")
        final_data_path = os.path.join(self.processed_data_path, "final.json")

        with open(subjective_data_path, "w", encoding="utf-8") as f:
            json.dump(subjective_data, f, ensure_ascii=False, indent=2)

        with open(general_data_path, "w", encoding="utf-8") as f:
            json.dump(general_data, f, ensure_ascii=False, indent=2)

        with open(final_data_path, "w", encoding="utf-8") as f:
            json.dump(epoch1_date, f, ensure_ascii=False, indent=2)

        logger.info(f"Subjective data saved to {subjective_data_path}")

    def run(self):
        subjective_messages_list = self.build_messages()
        responses = self.build_responses(subjective_messages_list)
        subjective_data = self.parse_subjective_data(subjective_messages_list, responses)
        logger.info(f"Subjective data processed, length: {len(subjective_data)}")
        subjective_length = len(subjective_data)

        general_data = self.load_final_general_data(sampled_num=subjective_length)
        logger.info(f"General data loaded, length: {len(general_data)}")
        self.save_data(subjective_data, general_data)

if __name__ == "__main__":
    data = Stage1Data()
    data.run()