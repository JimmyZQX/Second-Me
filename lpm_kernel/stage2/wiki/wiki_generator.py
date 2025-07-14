import json
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from lpm_kernel.stage2.wiki.base import EntityExtractor, PersonalWiki
from lpm_kernel.stage2.wiki.build_relation import RelationBuilder
from lpm_kernel.stage2.wiki.note_data_processor import note_processor

from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

def wiki_gen(preferred_language = "简体中文",
             note_remade_path: str = "resources/data/stage2/remade_note.json", old_version_data=None,
             entity_res_path: str = "resources/data/stage2/wiki/entity",
             relation_path: str = "resources/data/stage2/wiki/relations.json",
             wiki_path: str = "resources/data/stage2/wiki/wiki_res.json",
             mapping_output_path: str = "resources/data/stage2/wiki/mapping_output.json",):

    from lpm_kernel.base.database_operate import get_latest_global_bio, get_current_load

    global_bio = get_latest_global_bio().content_third_view
    current_load = get_current_load()
    user_name = current_load["name"]
    about_me = current_load["description"]

    note_processor(user_name, note_remade_path)

    with open(note_remade_path, "r") as f:
        data = json.load(f)

    logger.info(f"Initialization has finished, begin to extract entities from {len(data)} notes")

    input_data = {"notes": data, "userName": user_name, "globalBio": global_bio,
                  "preferredLanguage": preferred_language}

    entity_extractor = EntityExtractor()
    if old_version_data:
        old_entity_res_path = f"{old_version_data}/entity_res.json"
        with open(old_entity_res_path, 'r') as f:
            old_entity_result = json.load(f)
        input_data["entities"] = old_entity_result["entities"]
    entity_result = entity_extractor._call_(input_data)

    os.makedirs(os.path.dirname(entity_res_path), exist_ok=True)

    tmp_entity_path = f"{entity_res_path}/entity_res.json"
    os.makedirs(os.path.dirname(tmp_entity_path), exist_ok=True)

    with open(tmp_entity_path, "w") as f:
        if old_version_data:
            flag = "updated_entities"
        else:
            flag = "entities"
        json.dump(entity_result[flag], f, ensure_ascii=False, indent=4)
        logger.info(f"has extracted {len(entity_result[flag])}entities, begin to build relationship")

    if old_version_data:
        old_note = f"{old_version_data}/note_remade.json"
        with open(old_note, 'r') as f:
            old_data = json.load(f)
        data = old_data + data  # 合并新旧笔记
        with open(note_remade_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"After merged new and old notes, there are {len(data)}notes in total.")

    if old_version_data:
        total_entities_path = f"{entity_res_path}/entities_total.json"
        with open(total_entities_path, "w") as f:
            json.dump(entity_result["entities"], f, ensure_ascii=False, indent=4)

    if old_version_data:
        entities = entity_result["updated_entities"]
    else:
        entities = entity_result["entities"]

    entities = [entity for entity in entities if entity["genWiki"]]
    if not entities:
        logger.warning("没有有效的实体可供生成关系和wiki")
        return
    logger.info(f"There are {len(entities)} entities to build relationships.")

    relation = RelationBuilder(entities, data, relation_path)
    relation.build_relations()
    logger.info("has built relations, begin to generate wiki")

    entity_list = entities
    wiki = PersonalWiki()
    final_res = []

    def process_entity(entity):
        input_data = {
            "userName": user_name,
            "entityName": entity["name"],
            "wikiType": entity["entityType"],
            "timelines": entity["timelines"],
            "preferredLanguage": preferred_language,
            "aboutMe": about_me
        }
        res = wiki._call(input_data)
        wiki_text = res["entityWiki"]["wikiText"]

        timelines = entity["timelines"]
        note_ids = [int(timeline["noteId"]) for timeline in timelines]
        return {
            "entityName": entity["name"],
            "entityType": entity["entityType"],
            "description": wiki_text,
            "related_notes": note_ids,
        }

    with ThreadPoolExecutor() as executor:
        future_to_entity = {executor.submit(process_entity, entity): entity for entity in entity_list}
        for future in as_completed(future_to_entity):
            try:
                final_res.append(future.result())
            except Exception as exc:
                logger.error(f'Generated an exception: {exc}')
    # Add entity_id based on the order
    for idx, entity in enumerate(final_res):
        entity["entity_id"] = idx

    logger.info(f"has generated {len(final_res)} wikis, begin to map entity and wiki")
    # 构建实体mapping文件，用于后续任务
    mapping_res = []
    for entity in final_res:
        map_res = {
            "entity_name": entity["entityName"],
            "entity_description": entity["description"],
            "doc_id": entity["related_notes"],
            "entity_id": entity["entity_id"]
        }
        mapping_res.append(map_res)

    with open(wiki_path, "w") as f:
        json.dump(final_res, f, indent=4, ensure_ascii=False)

    with open(mapping_output_path, "w") as f:
        json.dump(mapping_res, f, indent=4, ensure_ascii=False)

    logger.info("all process has been done!")

if __name__ == "__main__":
    wiki_gen()
