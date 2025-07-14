import json
from concurrent.futures import ThreadPoolExecutor

from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService

load_dotenv(override=True)
user_llm_config_service = UserLLMConfigService()
user_llm_config = user_llm_config_service.get_available_llm()

if user_llm_config is None:
    OPENAI_API_KEY = None
    OPENAI_BASE_URL = None
    MODEL_NAME = None
else:
    OPENAI_API_KEY = user_llm_config.chat_api_key,
    OPENAI_BASE_URL = user_llm_config.chat_endpoint,

    MODEL_NAME = user_llm_config.chat_model_name


class RelationBuilder:
    def __init__(self, entities, notes, save_path):
        self.entities = entities
        self.notes = {note['noteId']: note for note in notes}
        self.save_path = save_path
        self.relation_map = []
        self.language = "简体中文"
        self.Relation_SYS = f"""
        你是一个关系描述生成器。你的任务是根据提供的笔记内容，生成两个实体之间的关系描述。
        请确保描述清晰、准确，并且能够反映出两个实体之间的联系。
        使用{self.language}输出。
        """

    def extract_related_notes(self):
        entity_notes_map = {}
        for entity in self.entities:
            entity_name = entity['name']
            related_notes = [int(timeline['noteId']) for timeline in entity['timelines']]
            entity_notes_map[entity_name] = related_notes
        return entity_notes_map

    def find_entity_relationships(self, entity_notes_map):
        entities = list(entity_notes_map.keys())
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                common_notes = set(entity_notes_map[entity1]).intersection(entity_notes_map[entity2])
                if common_notes:
                    self.relation_map.append({
                        'source': entity1,
                        'target': entity2,
                        'related_notes': list(common_notes)
                    })

    def get_note_content(self, note_id):
        note = self.notes.get(note_id, {})
        return note.get('content', note.get('insight', note.get('summary', '')))

    def format_context(self, related_notes):
        """
        Format the context information from related notes.
        """
        context = ""
        for note_id in related_notes:
            note_content = self.get_note_content(note_id)
            context += f"笔记 ID: {note_id}\n内容: {note_content}\n\n"
        return context

    def process_relation(self, source, target, related_notes):

        # Format the context from related notes
        context = self.format_context(related_notes)

        messages = [
            {"role": "system", "content": self.Relation_SYS},
            {"role": "user",
             "content": f"实体1: {source}\n实体2: {target}\n相关笔记:\n{context}\n请生成这两个实体之间的关系描述。"}
        ]
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            model = MODEL_NAME
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            return None

    def generate_relationship_descriptions(self):
        all_relations = self.relation_map
        if not all_relations:
            logger.warning("没有找到相互关联的实体")
            return

        max_workers = 10  # You can adjust this number based on your needs

        with ThreadPoolExecutor(max_workers=min(max_workers, len(all_relations))) as executor:
            # Use (index, future) structure to track each task's index
            futures = [(i, executor.submit(self.process_relation, relation['source'], relation['target'],
                                           relation['related_notes'])) for i, relation in enumerate(all_relations)]
            results = [None] * len(all_relations)  # Initialize a list to store results

            for i, future in tqdm(futures):
                try:
                    description = future.result()
                    results[i] = description  # Store the result at the correct index
                except Exception as e:
                    results[i] = f"Raise ERROR: {e} WHEN GENERATE RESPONSE"

        # Assign the descriptions back to the relation_map
        for i, relation in enumerate(self.relation_map):
            relation['description'] = results[i]
            relation['relationship_id'] = i

    def save_relations_to_json(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.relation_map, f, indent=4, ensure_ascii=False, )

    def build_relations(self):
        entity_notes_map = self.extract_related_notes()
        self.find_entity_relationships(entity_notes_map)
        self.generate_relationship_descriptions()
        self.save_relations_to_json(self.save_path)
