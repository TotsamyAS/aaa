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


def split_text_into_chunks(text, chunk_size=1024, overlap=128, sep="word"):
    chunks = []
    assert chunk_size >= 256, 'Error: chunk_size less than 256 not allowed!'
    # assert chunk_size > 2* overlap, 'Error: chunk_size must be at least x2 bigger than overlap!'
    assert sep in ['none', 'word'], 'Error: sep must be only "none" or "word"'
    print(
        f"Разбивка текста на чанки. Размер чанка: {chunk_size}, Перекрытие: {overlap}, Разбиение: {sep}")
    if sep == 'none':
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            chunks.append(text[start:end])
            start += chunk_size - overlap  # Сдвигаем начало следующего чанка с перекрытием

    elif sep == 'word':
        text = text.split(' ')  # разбиваем текст на слова
        chunk = ""  # токен из чанка
        curr_chunk_size = 0
        chunk_overlap = ""
        for word in text:
            if curr_chunk_size + len(" " + word) <= chunk_size:
                chunk = chunk + " " + word
                curr_chunk_size += len(" " + word)
                if len(chunk_overlap + " " + word) <= overlap:
                    chunk_overlap = chunk_overlap + " " + word
                # если длина перекрытия превышает нужную - добавляем слово, но въсрезаем то,что не влезло
                else:
                    chunk_overlap = " ".join(
                        chunk_overlap.split(" ")[1:]) + " " + word
                    chunk_overlap = chunk_overlap[chunk_overlap.find(
                        ' ', len(chunk_overlap)-overlap)+1:]
            else:
                chunks.append(chunk)
                chunk = chunk_overlap + " " + word
                curr_chunk_size = len(chunk_overlap + " " + word)
    else:
        print(
            'Что-то не так с сепаратором! Сепаратор может бы только: none, word')

    # print(f"Разбивка текста на чанки завершена")
    return chunks
