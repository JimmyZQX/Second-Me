import json
import random
from datetime import datetime, timedelta
from typing import Any

from lpm_kernel.file_data.document_repository import DocumentRepository


def note_processor(user_name, json_file_remade: str = "resources/data/stage2/remade_note.json"):
    doc_repository = DocumentRepository()
    documents = doc_repository.list()
    data_filtered = []
    data4wiki = []

    for item in documents:
        if item.mime_type not in ["WEBSITE", "DOCUMENT"]:
            add_info = {}
            image_template = []
            audio_template = []

            item.create_time = convert_to_east_eight_zone(item.create_time, return_datetime=True)

            short_audio_template = [f'{user_name}记录了如下数据:', f'{user_name}所记录的数据如下:',
                                    f'{user_name}记录的内容是:']

            insight = item.insight.get("insight")
            raw_content = item.raw_content

            if insight and raw_content:
                image_template = [
                    f'{user_name}记录了一张图片，主要是说:{raw_content}，图片里面的具体内容是: {insight}',
                    f'{user_name}拍摄了一张图片，其主要描述了:{raw_content}，图片详细呈现了: {insight}',
                    f'{user_name}记录了一张图片，其主要内容是:{raw_content}，具体内容为: {insight}'
                ]
                audio_template = [
                    f'{user_name}参与了一个线下讨论，讨论主题是:{raw_content}，主要内容是: {insight}',
                    f'{user_name}参与了一个关于{raw_content}的讨论，讨论的主要焦点是: {insight}',
                    f'{user_name}参加了一个关于{raw_content}的线下讨论，讨论的关键内容是: {insight}'
                ]

            if item.mime_type == "IMAGE":
                add_info['processed'] = random.choice(image_template)
            elif item.mime_type == "AUDIO":
                add_info['processed'] = random.choice(audio_template)
            else:
                content = item.raw_content
                title = item.title

                if item.mime_type:
                    add_info['processed'] = random.choice(short_audio_template) + str(content)
                else:
                    add_info['processed'] = random.choice(short_audio_template)
                    if title:
                        add_info['processed'] += "该数据主题是:" + title + "。更具体的内容为："
                    if content:
                        add_info['processed'] += str(content)

            wiki_item = {
                "noteId": item.id,
                "createTime": item.create_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "memoryType": item.mime_type,
                "content": item.raw_content,
                "insight": item.insight.get("insight"),
                "title": item.title,
                "summary": item.summary,
                "processed": add_info.get('processed')
            }
            data4wiki.append(wiki_item)
            data_filtered.append(item)

    with open(json_file_remade, 'w', encoding='utf-8') as file:
        json.dump(data4wiki, file, ensure_ascii=False, indent=4)


def convert_to_east_eight_zone(utc_time_str, return_datetime=False):
    if isinstance(utc_time_str, datetime):
        utc_time = utc_time_str
    else:
        try:
            time_format = "%Y-%m-%d %H:%M:%S.%f"
            utc_time = datetime.strptime(utc_time_str, time_format)
        except ValueError:
            try:
                time_format = "%Y-%m-%d %H:%M:%S"
                utc_time = datetime.strptime(utc_time_str, time_format)
            except ValueError:
                raise ValueError(f"无法解析时间格式: {utc_time_str}")
    east_eight_time = utc_time + timedelta(hours=8)

    if return_datetime:
        return east_eight_time
    else:
        return east_eight_time.strftime("%Y-%m-%d %H:%M:%S.%f")


if __name__ == '__main__':
    note_processor('RK')
