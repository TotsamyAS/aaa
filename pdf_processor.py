import pymupdf  # imports the pymupdf library


def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        # избавляемся от знаков переноса и переноса строк
    return text\
        .replace("-\n", "")\
        .replace("\n", "")


def split_text_into_chunks(text, chunk_size=1024, overlap=128):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start += chunk_size - overlap  # Сдвигаем начало следующего чанка с перекрытием
    return chunks
