from pymilvus import MilvusClient, DataType, utility, connections, Collection
from modules.pdf_processor import extract_text_from_pdf, split_text_into_chunks
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URI = "http://localhost:19530"
TOKEN = "root:Milvus"
# Подключение к Milvus
connections.connect("default", host="localhost", port="19530")

client = MilvusClient(uri=URI, token=TOKEN)
MODEL = (
    "cointegrated/rubert-tiny2"  # Название модели из HuggingFace Models
)
INFERENCE_BATCH_SIZE = 64  # Бэтчи вывода модели

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)


def get_embeddings(texts):
    """
    Получает эмбеддинги для списка текстов.
    """
    # Токенизация текстов
    inputs = tokenizer(texts, padding=True, truncation=True,
                       return_tensors="pt", max_length=512)

    # Получение скрытых состояний от модели
    with torch.no_grad():
        outputs = model(**inputs)

    # Усреднение скрытых состояний по всем токенам для получения эмбеддинга
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings


def initialize_collections():
    """
    Инициализация коллекций в Milvus, если они не существуют.
    """
    # Проверяем, существует ли коллекция для чанков текста
    if not utility.has_collection("document_db_collection"):
        chunk_schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        chunk_schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True)
        chunk_schema.add_field(field_name="doc_name",
                               datatype=DataType.VARCHAR, max_length=128)
        chunk_schema.add_field(field_name="text_embedding",
                               datatype=DataType.FLOAT_VECTOR, dim=312)
        chunk_schema.add_field(field_name="text_content",
                               datatype=DataType.VARCHAR, max_length=2048)  # макс. длина в 2 раза больше чанка, т.к. кириллица имеет кодировку в 2 байта, тогда 512 символов будет не превышать 1024 символа из utf-8

        chunk_index_params = client.prepare_index_params()
        chunk_index_params.add_index(
            field_name="text_embedding", index_type="AUTOINDEX", metric_type="COSINE")

        client.create_collection(
            collection_name="document_db_collection",
            schema=chunk_schema,
            index_params=chunk_index_params,
        )
        logger.info(f" Коллекция 'document_db_collection' создана.")

    # Проверяем, существует ли коллекция для названий документов
    if not utility.has_collection("document_name_collection"):
        name_schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        name_schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True)
        name_schema.add_field(field_name="doc_name",
                              datatype=DataType.VARCHAR, max_length=128)
        name_schema.add_field(field_name="date_added",  # Добавляем поле для даты
                              datatype=DataType.VARCHAR, max_length=10)  # Формат ДД.ММ.ГГГГ
        # Добавляем фиктивное векторное поле
        name_schema.add_field(field_name="dummy_vector",
                              datatype=DataType.FLOAT16_VECTOR, dim=2)

        name_index_params = client.prepare_index_params()
        name_index_params.add_index(
            field_name="doc_name", index_type="AUTOINDEX")
        name_index_params.add_index(
            field_name="dummy_vector", index_type="AUTOINDEX")

        client.create_collection(
            collection_name="document_name_collection",
            schema=name_schema,
            index_params=name_index_params,
        )
        logger.info(f" Коллекция 'document_name_collection' создана.")


def drop_collections():
    client.drop_collection("document_db_collection")
    client.drop_collection("document_name_collection")


def upload_pdf_to_db(pdf_path, doc_name, date_added):
    """
    Загружает PDF-файл в базу данных.
    """

    # Извлекаем текст из PDF
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)

    # Генерируем эмбеддинги для каждого чанка
    embeddings = get_embeddings(chunks)

    # Сохраняем название документа в document_name_collection
    client.insert(
        collection_name="document_name_collection",
        data=[{"doc_name": doc_name, "date_added": date_added,
               "dummy_vector": np.array(
                   [0.0, 0.0], dtype=np.float16)}],
    )

    # Сохраняем чанки и их эмбеддинги в document_db_collection
    data = []
    for chunk, embedding in zip(chunks, embeddings):
        data.append({
            "doc_name": doc_name,
            "text_embedding": embedding.tolist(),
            "text_content": chunk,
        })

    client.insert(
        collection_name="document_db_collection",
        data=data,
    )

    logger.info(f"Документ '{doc_name}' успешно загружен в базу данных.")


def delete_pdf_from_db(doc_name):
    # Проверяем, существует ли документ в document_name_collection
    response = client.query(
        collection_name="document_name_collection",
        filter=f"doc_name == '{doc_name}'",
        output_fields=["doc_name"],
    )
    if not response:
        logger.info(f" Документ '{doc_name}' не найден.")
        return

    # Удаляем все записи с совпадающим doc_name
    client.delete(
        collection_name="document_db_collection",
        filter=f"doc_name == '{doc_name}'"
    )
    logger.info(
        f"Документ '{doc_name}' успешно удалён из базы данных 'document_db_collection'.")
    client.delete(
        collection_name="document_name_collection",
        filter=f"doc_name == '{doc_name}'"
    )
    logger.info(
        f"Документ '{doc_name}' успешно удалён из базы данных 'document_name_collection'.")


def get_uploaded_documents(limit=1000, offset=0):
    """
    Возвращает список загруженных документов с их именами и датами добавления.
    """
    response = client.query(
        collection_name="document_name_collection",
        filter="",
        offset=offset,  # Смещение
        limit=limit,
        output_fields=["doc_name", "date_added"],  # Запрашиваем имя и дату
    )
    return [{"doc_name": item["doc_name"], "date_added": item["date_added"]} for item in response]


def semantic_search(query, top_k=3):
    pass
