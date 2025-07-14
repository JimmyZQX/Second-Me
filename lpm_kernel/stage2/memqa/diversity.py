import os
import sys

sys.path.append(os.getcwd().rsplit('/', 1)[0])
from lpm_kernel.base.data import BaseData
from difflib import SequenceMatcher
from lpm_kernel.base.stage2_prompt import DIVERSITY_QUESTION_SYSTEM_PROMPT, DIVERSITY_ANSWER_SYSTEM_PROMPT
import os, re, json, random
from lpm_kernel.base.database_operate import get_latest_global_bio
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

class templater:
    def __init__(self,
                 q_dict,
                 a_dict,
                 user_name='',
                 global_bio='',
                 status_bio=''
                 ):
        self.a_dict = a_dict
        self.q_dict = q_dict
        self.user_name = user_name
        self.global_bio = global_bio
        self.status_bio = status_bio
        self.a_temp = DIVERSITY_ANSWER_SYSTEM_PROMPT

    def get_A_template(self, question_type):
        templ = self.a_temp
        answer_rule = ''
        required_type = self.q_dict[question_type]['requiredAnswerTypes']
        optional_type = self.q_dict[question_type]['optionalAnswerTypes']
        if required_type:
            answer_rule = 'The required expressions to be included in the response:\n'
            for ind, answer_type in enumerate(required_type):
                sub_prompt = self.a_dict[answer_type]['prompt']
                answer_rule += f'{ind + 1}. {sub_prompt}\n'
        if optional_type:
            k = random.randint(1, len(optional_type))
            chosen_optional_type = random.sample(optional_type, k)
        else:
            chosen_optional_type = []
        if chosen_optional_type:
            answer_rule += 'The optional, combinable response expression:\n'
            for ind, answer_type in enumerate(chosen_optional_type):
                sub_prompt = self.a_dict[answer_type]['prompt']
                answer_rule += f'{ind + 1}. {sub_prompt}\n'
        templ = templ.replace('__answer_rule__', answer_rule)

        # 检查是否需要结合bio
        bio = ''
        status_bio_flag = False
        global_bio_flag = False
        for type in chosen_optional_type:
            extra_info = self.a_dict[type]['extraInfo']
            if 'statusBio' in extra_info:
                status_bio_flag = True
                break
            if 'globalBio' in extra_info:
                global_bio_flag = True
                break
        if status_bio_flag:
            bio += f'Your recent status is:\n\n{self.status_bio}\n'
        if global_bio_flag:
            bio += f'Your profile is:\n\n{self.global_bio}\n'

        if bio:
            bio += 'You may refer to the above information when responding, but do not overuse it.'
            templ = templ.replace('# Guidelines #', f'# Guidelines #\n{bio}')

        return templ, chosen_optional_type

    def get_Q_template(self, question_type_prompt):

        return DIVERSITY_QUESTION_SYSTEM_PROMPT.replace('__question_type__', question_type_prompt)


class DiversityData(BaseData):
    def __init__(self, is_cot: bool = True, max_workers=10, language='简体中文', user_name='user'):
        super().__init__(is_cot=is_cot, max_workers=max_workers)
        self.USERNAME = user_name
        self.language_desc = f'Keep your response in {language}'
        self.entity2desc, self.entity2type, self.QA_config = self.preprocess()
        self.q_dict = {item['type']: {k: item[k] for k in item if k != 'type'} for item in self.QA_config['query']}
        self.a_dict = {item['type']: {k: item[k] for k in item if k != 'type'} for item in self.QA_config['answer']}
        self.global_bio, self.status_bio = self.get_bio()
        self.templater = templater(self.q_dict, self.a_dict, self.USERNAME, self.global_bio, self.status_bio)

    def get_bio(self):
        global_bio = get_latest_global_bio().content_third_view
        status_bio = []
        return global_bio, status_bio

    def preprocess(self, config_path: str = "lpm_kernel/base/config_diversity.json",
                   wiki: str = "resources/data/stage2/wiki/wiki_res.json",
                   entities_path: str = "resources/data/stage2/wiki/mapping_output.json",
                   note_path: str = "resources/data/stage2/remade_note.json"):

        with open(wiki, 'r', encoding='utf-8') as file:
            wiki = json.load(file)
        entity2type = {item['entityName']: item['entityType'] for item in wiki}

        with open(entities_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
            entity2desc = {item['entity_name']: {key: value for key, value in item.items() if key != "entity_name"} for
                           item
                           in entities}

        with open(note_path, 'r', encoding='utf-8') as f:
            note_data = json.load(f)
            id2note = {item["noteId"]: {key: value for key, value in item.items() if key != "noteId"} for item in
                       note_data}

        for entity, entity_info in entity2desc.copy().items():
            doc_ids = entity_info['doc_id']
            tmp = []
            for doc_id in doc_ids:
                note_desc = id2note.get(doc_id, '')
                if note_desc:
                    tmp.append(note_desc)
            entity2desc[entity]['note'] = tmp

        entity2desc.pop(self.USERNAME, None)
        entity2desc.pop(self.USERNAME.upper(), None)

        # 排除包含时间格式的键
        time_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        filtered_data = {k: v for k, v in entity2desc.items() if not re.match(time_pattern, k)}
        entity2desc = filtered_data

        # note级别数据清理
        for entity, entity_info in entity2desc.copy().items():
            clusters = entity_info['note']
            unique_dicts, cnt = self.remove_similar_dicts(clusters, similarity_threshold=0.9)
            entity2desc[entity]['note'] = unique_dicts

        # 读取config文件
        with open(config_path, 'r', encoding='utf-8') as f:
            QA_config = json.load(f)

        return entity2desc, entity2type, QA_config

    def string_similarity(self, str1, str2):
        """计算两个字符串的编辑距离相似度"""
        return SequenceMatcher(None, str1, str2).ratio()

    def remove_similar_dicts(self, dict_list, similarity_threshold=0.6):
        """剔除content字段相似度大于阈值的字典"""
        unique_dicts = []
        cnt = 0
        for i, current_dict in enumerate(dict_list):
            if not current_dict['content']:
                continue
            is_similar = False
            for j in range(len(unique_dicts)):
                if not unique_dicts[j]['content']:
                    continue
                if self.string_similarity(current_dict['content'], unique_dicts[j]['content']) > similarity_threshold:
                    is_similar = True
                    logger.info(
                        f" {current_dict['content'][-100:]}\n is similar to: \n{unique_dicts[j]['content'][-100:]}\n____________________")
                    cnt += 1
                    break

            if not is_similar:
                unique_dicts.append(current_dict)

        return unique_dicts, cnt

    def cluster_clean(self, chunks, update_flag=False):
        if not update_flag and os.path.exists('./raw_data/cleaned_clusters.json'):
            with open('./raw_data/cleaned_clusters.json', 'r', encoding='utf-8') as f:
                return json.load(f)

        clusters = []
        for cluster_inds in chunks['cluster_ids'].values():
            cluster_chunks = []
            for ind in cluster_inds:
                chunk = chunks['chunks'][int(ind)]
                cluster_chunks.append(chunk)
            clusters.append(cluster_chunks)
        # 清除同一个note来源的chunk数据，只保留最多一个
        dedup_cnt = 0
        processed_clusters = []
        for cluster in clusters:
            seen_ids = set()
            unique_data = []
            for item in cluster:
                if item['document_id'] not in seen_ids:
                    unique_data.append(item)
                    seen_ids.add(item['document_id'])
                else:
                    dedup_cnt += 1
            processed_clusters.append(unique_data)
        logger.info(f"Note level dedup count: {dedup_cnt}")

        # 对每个clusters下，进行chunk级别的去重
        final_clusters = []
        tol_cnt = 0
        for processed_cluster in processed_clusters:
            processed_cluster, cnt = self.remove_similar_dicts(processed_cluster)
            final_clusters.append(processed_cluster)
            tol_cnt += cnt
        logger.info(f"Chunk level dedup count: {tol_cnt}")

        with open('./raw_data/cleaned_clusters.json', 'w', encoding='utf-8') as f:
            json.dump(final_clusters, f, ensure_ascii=False, indent=4)

        return final_clusters

    def preprocess_chunks(self, chunks):
        del_str = ['This chunk is about DOCUMENT:', 'This chunk is about WEBSITE:',
                   'This is the information of an Audio:', 'This chunk is about file:']
        for sub_dict in chunks['chunks']:
            for del_str_item in del_str:
                sub_dict['content'] = sub_dict['content'].replace(del_str_item, '')
        return chunks

    def build_question_messages(self, clusters, question_types):
        question_messages_list = []
        # 1 get user content
        for cluster, question_type in zip(clusters, question_types):
            entity = cluster['entity_name']
            entity_desc = cluster['entity_description']
            entity_desc = f"Entity'{entity}'：{entity_desc}"
            tmpl = f""""For {entity_desc}, here is your relevant content:\n"""
            chunk_tmpl = ''
            for ind, entity_dict in enumerate(cluster['note']):
                if 'processed' in entity_dict:
                    content = entity_dict['processed']
                else:
                    content = entity_dict['content']
                    title = entity_dict['title']
                    insight = entity_dict['insight']
                    content = f"Title: {title}\nYour content: {content}\n Content insight: {insight}"

                tmp = f"# Content {ind + 1} #\n{content}\n"
                chunk_tmpl += tmp
            user_content = tmpl + chunk_tmpl
            messages = [
                {"role": "system",
                 "content": self.templater.get_Q_template(question_type_prompt=self.q_dict[question_type]['prompt'])},
                {"role": "user", "content": user_content + self.language_desc}]
            question_messages_list.append(messages)
        return question_messages_list

    def build_answer_messages(self, clusters, questions, question_types):
        answer_messages_list, answer_type_list = [], []
        for cluster, question_list, question_type in zip(clusters, questions, question_types):
            entity = cluster['entity_name']
            entity_desc = cluster['entity_description']
            entity_desc = f"Following is the memory:\nEntity'{entity}',Relevant Info：'{entity_desc}'"

            tmpl = f"""Regarding {entity_desc}, here is the relevant information you previously mentioned:\n\n"""

            chunk_tmpl = ''
            for ind, entity_dict in enumerate(cluster['note']):
                if 'processed' in entity_dict:
                    content = entity_dict['processed']
                else:
                    content = entity_dict['content']
                    title = entity_dict['title']
                    insight = entity_dict['insight']
                    content = f"Title: {title}\nYour content: {content}\nContent insight: {insight}"

                tmp = f"___________________\n{content}\n"
                chunk_tmpl += tmp
            for question in question_list:
                user_content = tmpl + chunk_tmpl
                system_prompt, answer_type = self.templater.get_A_template(question_type)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content + self.language_desc}]
                answer_messages_list.append(messages)
                answer_type_list.append(answer_type)
        return answer_messages_list, answer_type_list

    def build_question_response(self, question_messages_list, explode_clusters, explode_questions_types):
        question_responses = self.build_responses(question_messages_list)
        questions_list, flat_clusters_list, flat_questions_types_list = [], [], []
        for question_response, cluster, question_type in zip(question_responses, explode_clusters,
                                                             explode_questions_types):
            questions = []
            try:
                last_think_index = question_response.rfind("</think>")
                question_response = question_response if last_think_index == -1 else question_response[
                                                                                     last_think_index + len(
                                                                                         "</think>"):]
                questions = question_response.split('|||')
            except Exception as e:
                logger.error(f"{e}")
                questions = []
            if questions:
                try:
                    # 处理包含 '|' 的情况
                    if '|' in questions[0]:
                        questions = questions[0].split('|')
                except (IndexError, TypeError):
                    pass  # 避免索引错误或类型错误

                # 去除首尾空白和标签，并做安全处理
                questions = [
                    question.strip().replace("<answer>", "").replace("</answer>", "")
                    for question in questions
                    if question is not None  # 排除 None 元素
                ]

            flat_clusters_list.extend([cluster] * len(questions))
            flat_questions_types_list.extend([question_type] * len(questions))
            questions_list.append(questions)
        return questions_list, flat_clusters_list, flat_questions_types_list

    def build_answer_response(self, answer_messages_list):
        answer_responses = self.build_responses(answer_messages_list)
        return answer_responses

    def pipline(self, clusters, output_path, aug_para, filter_flag=False):
        explode_clusters = []
        explode_questions_types = []
        for item in clusters:
            # 添加元素三次
            explode_clusters.extend([item] * aug_para)
            # 随机选择三个不同的数字
            weights = [v['weight'] for v in self.q_dict.values()]
            random_types = random.choices(list(self.q_dict.keys()), weights, k=aug_para)
            explode_questions_types.extend(random_types)
        # 1 build question messages and answer responses
        question_messages_list = self.build_question_messages(explode_clusters, explode_questions_types)
        question_list, flat_clusters_list, flat_questions_types_list = self.build_question_response(
            question_messages_list, explode_clusters, explode_questions_types)
        # 2 build answer messages and answer responses
        answer_messages_list, answer_type_list = self.build_answer_messages(explode_clusters, question_list,
                                                                            explode_questions_types)
        answer_list = self.build_answer_response(answer_messages_list)

        final_question_list = []
        for question in question_list:
            final_question_list.extend(question)

        logger.info(f"flat_clusters_list len: {len(flat_clusters_list)}")
        logger.info(f"final_question_list len: {len(final_question_list)}")
        logger.info(f"answer_list len: {len(answer_list)}")
        logger.info(f"flat_questions_types_list len: {len(flat_questions_types_list)}")
        logger.info(f"answer_type_list len: {len(answer_type_list)}")
        # 存储数据
        data = []
        for cluster, question, answer, question_type, answer_type in (
                zip(flat_clusters_list, final_question_list, answer_list, flat_questions_types_list, answer_type_list)):
            # 检查 question 和 answer 是否为字符串类型
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
            if len(question) == 0 or len(answer) == 0:
                continue
            data.append({
                'user': question,
                'assistant': answer,
                'entity_name': cluster['entity_name'],
                'question_type': question_type,
                'answer_type': answer_type,
                'doc_id': cluster['doc_id'],
            })

        return data

    def run(self, output_path: str = "resources/data/stage2/processed/diversity.json"):
        entity2desc_list = [{**{"entity_name": k}, **v} for k, v in self.entity2desc.items()]

        base, ext = os.path.splitext(output_path)
        # 构造新的路径
        large_path = f"{base}_large{ext}"
        mini_path = f"{base}_mini{ext}"
        tiny_path = f"{base}_tiny{ext}"

        # global问题，只处理大于2的, 对于特别大的进行拆分
        large_clusters = [item for item in entity2desc_list if len(item['note']) >= 8]

        exploded_clusters = []
        # 拆分
        for sub_dict in large_clusters:
            for i in range(0, len(sub_dict['note']), 4):
                tmp_dict = sub_dict.copy()

                tmp_dict['note'] = sub_dict['note'][i:i + 4]
                tmp_dict['doc_id'] = sub_dict['doc_id'][i:i + 4]
                exploded_clusters.append(tmp_dict)

            # 为保证全局效果，再添加一些大的global数据
            notes_and_ids = list(zip(sub_dict['note'], sub_dict['doc_id']))
            for _ in range(len(sub_dict['note']) // 10 + 1):
                tmp_dict = sub_dict.copy()
                sampled_notes_and_ids = random.sample(notes_and_ids, min(10, len(notes_and_ids)))
                tmp_dict['note'], tmp_dict['doc_id'] = zip(*sampled_notes_and_ids)  # 解压缩成两个列表
                exploded_clusters.append(tmp_dict)

        # 处理小cluster
        mini_clusters = [item for item in entity2desc_list if len(item['note']) < 8 and len(item['note']) > 1]

        # 处理其余cluster
        tiny_clusters = [item for item in entity2desc_list if len(item['note']) <= 1]
        filtered_tiny_clusters = [d for d in tiny_clusters if
                                  self.entity2type.get(d["entity_name"], '') in ['PERSON', 'LOCATION', 'CONCEPT',
                                                                                 'NORMAL_ENTITY']]

        logger.info('执行大cluster生成')
        data_large = self.pipline(exploded_clusters, large_path, 4)

        logger.info('执行小cluster生成')
        data_mini = self.pipline(mini_clusters, mini_path, 3)

        data_tiny = self.pipline(filtered_tiny_clusters, tiny_path, 2)

        combined_list = data_large + data_mini + data_tiny

        total_entries = len(combined_list)
        logger.info(total_entries)

        with open(os.path.join(output_path), 'w', encoding='utf-8') as f:
            json.dump(combined_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    diversity_qa = DiversityData()
    diversity_qa.run()
