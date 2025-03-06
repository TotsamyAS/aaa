import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import logging
import time
# Импортируем функции из database.py
from modules.database import get_embeddings, search_documents

# Инициализация логирования - для проверки работы модели
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Определяем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Используемое устройство: {device}")

# Конфигурация модели
MODEL_NAME = "IlyaGusev/saiga_mistral_7b_lora"

# Системный промпт
DEFAULT_SYSTEM_PROMPT = (
    "Ты — AAA (Augmented Artificial Assistant), русскоязычный автоматический ассистент. "
    "Ты разговариваешь с людьми и помогаешь им находить информацию в документах. "
    "Используй в первую очередь факты из предоставленных документов для ответа на вопросы."
)

# Загрузка модели и токенизатора
try:
    logger.info("Начало загрузки модели...")
    config = PeftConfig.from_pretrained(MODEL_NAME)
    logger.info("Конфигурация модели загружена.")

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_8bit=device == "cuda",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    logger.info("Базовая модель загружена.")

    model = PeftModel.from_pretrained(
        model,
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    logger.info("LoRA-адаптеры загружены.")

    model.eval()
    logger.info("Модель переведена в режим eval.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    logger.info("Токенизатор загружен.")

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    logger.info("Конфигурация генерации загружена.")

    logger.info("Модель успешно загружена.")
    model_ready = True
except Exception as e:
    logger.info(f"Ошибка при загрузке модели: {e}")

# Класс для управления диалогом


class Conversation:
    def __init__(self):
        self.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self):
        prompt = ""
        for message in self.messages:
            if message["role"] == "system":
                prompt += f"<s>system\n{message['content']}</s>"
            elif message["role"] == "user":
                prompt += f"<s>user\n{message['content']}</s>"
            elif message["role"] == "bot":
                prompt += f"<s>bot\n{message['content']}</s>"
        prompt += "<s>bot\n"
        return prompt.strip()


# Глобальная переменная для хранения диалога
conversation = Conversation()


# Функция для генерации ответа без RAG
def generate_response(user_message):
    if not model_ready:
        logger.error("Модель не загружена.")
        return "Модель не загружена. Пожалуйста, проверьте ошибки."
    logger.info(f"Получен запрос от пользователя: {user_message}")

    # Добавляем сообщение пользователя в диалог
    conversation.add_user_message(user_message)
    logger.info("Сообщение пользователя добавлено в диалог.")

    # Формируем промпт для модели
    prompt = conversation.get_prompt()
    logger.info("Промпт для модели сформирован.")

    # Генерируем ответ с помощью модели
    start_time = time.time()
    logger.info("Начало генерации ответа...")
    inputs = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=False).to(device)
    output_ids = model.generate(
        **inputs, generation_config=generation_config
    )[0]
    output = tokenizer.decode(
        output_ids[len(inputs["input_ids"][0]):], skip_special_tokens=True)
    eval_time = time.time() - start_time
    logger.info(
        f"Генерация ответа заняла {eval_time//60} минут, {eval_time - eval_time//60} секунд.")
    logger.info(f"Ответ модели сгенерирован: {output}")

    # Добавляем ответ бота в диалог
    conversation.add_bot_message(output)
    logger.info("Ответ бота добавлен в диалог.")

    return output

# TODO: Функция для генерации ответа с RAG


def generate_response_with_RAG(user_message):
    # Векторизация запроса пользователя
    query_embedding = get_embeddings([user_message]).cpu().numpy()

    # Поиск релевантных документов
    # Ищем 3 наиболее релевантных документа
    relevant_docs = search_documents(query_embedding, top_k=3)

    # Формируем контекст для модели
    context = "\n".join(relevant_docs)
    prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\nКонтекст:\n{context}\n\nВопрос: {user_message}\nОтвет:"

    # Генерируем ответ с помощью модели
    inputs = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=False).to(device)
    output_ids = model.generate(
        **inputs, generation_config=generation_config)[0]
    output = tokenizer.decode(
        output_ids[len(inputs["input_ids"][0]):], skip_special_tokens=True)

    # Добавляем ответ бота в диалог
    conversation.add_bot_message(output)

    return output
