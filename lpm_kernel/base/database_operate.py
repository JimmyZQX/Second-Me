from datetime import datetime
from typing import List, Optional

import numpy as np

from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.file_data.document_service import document_service
from lpm_kernel.models.l1 import L1Bio, L1Shade, L1Version, L1Cluster
from lpm_kernel.models.l1 import StatusBioDTO, GlobalBioDTO
from lpm_kernel.models.load import Load
from lpm_kernel.models.status_biography import StatusBiography
from lpm_kernel.stage2.bio import Bio
from lpm_kernel.stage2.bio import Note, Chunk

logger = get_train_process_logger()


def store_version(
        session, new_version_number: int, description: str = None
) -> L1Version:
    """Store stage2 version information"""
    version = L1Version(
        version=new_version_number,
        create_time=datetime.now(),
        status="active",
        description=description or f"stage2 data version {new_version_number}",
    )
    session.add(version)
    return version


def store_bio(session, new_version: int, bio_data: str) -> None:
    """Store Bio data"""
    if not bio_data:
        logger.warning("No bio data found")
        return

    bio_record = L1Bio(
        version=new_version,
        content=bio_data,
        content_third_view=bio_data,
        summary=bio_data,
        summary_third_view=bio_data,
        create_time=datetime.now(),
    )
    session.add(bio_record)


def store_clusters(session, new_version: int, cluster_list: list) -> None:
    """Store Clusters data"""
    if not cluster_list:
        logger.warning("No clusters data found")
        return

    for cluster in cluster_list:
        cluster_data = L1Cluster(
            version=new_version,
            cluster_id=cluster.get("clusterId"),
            memory_ids=[m.get("memoryId") for m in cluster.get("memoryList", [])],
            cluster_center=cluster.get("clusterCenter"),
            create_time=datetime.now(),
        )
        session.add(cluster_data)


def store_shades(session, new_version: int, shades_list: list) -> None:
    """Store Shades data"""
    if not shades_list:
        logger.warning("No shades data found")
        return

    for shade in shades_list:
        shade_data = L1Shade(
            version=new_version,
            name=shade.get("name"),
            aspect=shade.get("aspect"),
            icon=shade.get("icon"),
            desc_third_view=shade.get("descThirdView"),
            content_third_view=shade.get("contentThirdView"),
            desc_second_view=shade.get("descSecondView"),
            content_second_view=shade.get("contentSecondView"),
            create_time=datetime.now(),
        )
        session.add(shade_data)


def get_current_load() -> dict:
    """获取当前 load 记录

    Returns:
        dict: 包含 load 数据的字典，如果没有找到则返回空字典
    """
    try:
        with DatabaseSession.session() as session:
            current_load = session.query(Load).order_by(Load.created_at.desc()).first()

            if not current_load:
                logger.warning("数据库中未找到 load 记录")
                return {}
            load_dict = {
                "id": current_load.id,
                "name": current_load.name,
                "email": current_load.email,
                "description": current_load.description,
            }

            return load_dict
    except Exception as e:
        logger.error(f"获取当前 load 时出错: {str(e)}", exc_info=True)
        return {}


def get_latest_global_bio() -> Optional[GlobalBioDTO]:
    """Get the latest global biography in third-person view

    Returns:
        str: Third-person view content of global biography, or None if not found
    """
    try:
        with DatabaseSession.session() as session:
            # Get the latest version of stage2 data
            latest_version = (
                session.query(L1Version).order_by(L1Version.version.desc()).first()
            )

            if not latest_version:
                logger.warning("No L1Version found in database")
                return None
            bio = (
                session.query(L1Bio)
                .filter(L1Bio.version == latest_version.version)
                .first()
            )

            if not bio:
                logger.warning("No L1Bio found in database")
                return None

            return GlobalBioDTO.from_model(bio)
    except Exception as e:
        logger.error(f"Error getting global biography: {str(e)}", exc_info=True)
        return None


def extract_notes_from_documents(documents) -> tuple[List[Note], list]:
    """Extract Note objects and memory list from documents

    Args:
        documents: Document list containing L0 data

    Returns:
        tuple: (notes_list, memory_list)
            - notes_list: List of Note objects
            - memory_list: List of memory dictionaries for clustering
    """
    notes_list = []
    memory_list = []

    for doc in documents:
        doc_id = doc.get("id")
        doc_embedding = document_service.get_document_embedding(doc_id)
        chunks = document_service.get_document_chunks(doc_id)
        all_chunk_embeddings = document_service.get_chunk_embeddings_by_document_id(
            doc_id
        )

        if not doc_embedding:
            logger.warning(f"Document {doc_id} missing document embedding")
            continue
        if not chunks:
            logger.warning(f"Document {doc_id} missing chunks")
            continue
        if not all_chunk_embeddings:
            logger.warning(f"Document {doc_id} missing chunk embeddings")
            continue

        # Ensure create_time is in string format
        create_time = doc.get("create_time")
        if isinstance(create_time, datetime):
            create_time = create_time.strftime("%Y-%m-%d %H:%M:%S")

        # Get document insight and summary
        insight_data = doc.get("insight", {})
        summary_data = doc.get("summary", {})

        if insight_data is None:
            insight_data = {}
        if summary_data is None:
            summary_data = {}

        # Build Note object
        note = Note(
            noteId=doc_id,
            content=doc.get("raw_content", ""),
            createTime=create_time,
            memoryType="TEXT",
            embedding=np.array(doc_embedding),
            chunks=[
                Chunk(
                    id=chunk.id,
                    document_id=doc_id,
                    content=chunk.content,
                    embedding=np.array(all_chunk_embeddings.get(chunk.id))
                    if all_chunk_embeddings.get(chunk.id)
                    else None,
                    tags=chunk.tags if hasattr(chunk, "tags") else None,
                    topic=chunk.topic if hasattr(chunk, "topic") else None,
                )
                for chunk in chunks
                if all_chunk_embeddings.get(chunk.id)
            ],
            title=insight_data.get("title", ""),
            summary=summary_data.get("summary", ""),
            insight=insight_data.get("insight", ""),
            tags=summary_data.get("keywords", []),
        )
        notes_list.append(note)
        memory_list.append({"memoryId": str(doc_id), "embedding": doc_embedding})

    return notes_list, memory_list
