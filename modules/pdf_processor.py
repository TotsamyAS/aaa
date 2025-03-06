import pymupdf  # imports the pymupdf library

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    logger.info(f"PDF {pdf_path} Получен. Начинается извлечение...")
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        # избавляемся от знаков переноса и переноса строк
    logger.info(f"Извлечение PDF {pdf_path} Закончено")
    return text\
        .replace("-\n", "")\
        .replace("\n", "")


def split_text_into_chunks(text, chunk_size=1024, overlap=128):
    chunks = []
    start = 0
    logger.info(
        f"Разбивка текста на чанки. Размер чанка: {chunk_size}, Перекрытие: {overlap}")
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start += chunk_size - overlap  # Сдвигаем начало следующего чанка с перекрытием
    logger.info(f"Разбивка текста на чанки завершена")
    return chunks

# если буду разбивать текст по словам, необходимо помнить о том, что нужно считать длину текста в кодировке UTF-8
    # for chunk in chunks:
    #     logger.info(
    #         f"Длина чанка: {len(chunk)} - нужно 1024  | А в UTF-8: {len(chunk.encode('utf-8'))}")
