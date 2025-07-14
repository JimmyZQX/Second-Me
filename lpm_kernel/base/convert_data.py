import argparse
import json
import os
import random

from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def convert_standard_data():
    final_path = "resources/data/merged.json"
    final_data = []
    stage1_data, stage2_data, stage3_data = [], [], []
    pipeline_dir = "resources/data"
    stage1_data_path = os.path.join(pipeline_dir, "stage1")
    stage2_data_path = os.path.join(pipeline_dir, "stage2", "processed")
    stage3_data_path = os.path.join(pipeline_dir, "stage3")

    # stage1 格式正确，直接读取即可
    stage1_data_list = ["subjective.json", "general.json"]
    stage1_total = 0
    for data_file in stage1_data_list:
        data = load_json(os.path.join(stage1_data_path, data_file))
        file_len = len(data)
        logger.info(f"[Stage1] File: {data_file}, Count: {file_len}")
        stage1_data.extend(data)
        stage1_total += file_len
    logger.info(f"[Stage1] Total Count: {stage1_total}")

    stage2_data_list = ["subjective_entity.json", "subjective_description.json",
                        "subjective_relation.json", "stage2_qa.json", "diversity.json"]
    stage2_total = 0
    for data_file in stage2_data_list:
        data_path = os.path.join(stage2_data_path, data_file)
        if not os.path.exists(data_path):
            logger.warning(f"[Stage2] File not found: {data_file}")
            continue
        data = load_json(data_path)
        if data_file == "stage2_qa.json":
            data = [{"user": qa["query"], "assistant": qa["answer"]} for qa in data]
        elif data_file == "diversity.json":
            data = [{"user": qa["user"], "assistant": qa["assistant"]} for qa in data]
        file_len = len(data)
        logger.info(f"[Stage2] File: {data_file}, Count: {file_len}")
        stage2_data.extend(data)
        stage2_total += file_len
    logger.info(f"[Stage2] Total Count: {stage2_total}")

    stage3_data_list = ["synthetic_data_no_notes_answers.json", "synthetic_data_with_notes_answers.json"]
    stage3_total = 0
    for data_file in stage3_data_list:
        data_path = os.path.join(stage3_data_path, data_file)
        if not os.path.exists(data_path):
            logger.warning(f"[Stage3] File not found: {data_file}")
            continue
        data = load_json(data_path)
        data = [{"user": qa["generated_questions"], "assistant": qa["answer"]} for qa in data if
                "Error generating data" not in qa["answer"]]
        file_len = len(data)
        logger.info(f"[Stage3] File: {data_file}, Count: {file_len}")
        stage3_data.extend(data)
        stage3_total += file_len
    logger.info(f"[Stage3] Total Count: {stage3_total}")

    logger.info("[Stage All] Shuffle")
    final_data.extend(stage1_data)
    final_data.extend(stage2_data)
    final_data.extend(stage3_data)
    random.shuffle(final_data)


    final_data = [item for item in final_data if item.get("assistant") is not None]
    logger.info(f"[Final] Total Count: {len(final_data)}")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    convert_standard_data()
