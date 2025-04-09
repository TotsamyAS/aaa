import logging
import time
from pathlib import Path
import os
from llama_cpp import Llama, llama_log_set
from ctypes import CFUNCTYPE, c_void_p, c_int

# import llama_cpp

# Импортируем функции из database.py
from modules.database import get_embeddings, semantic_search

# Функция-пустышка для логов


@CFUNCTYPE(None, c_int, c_void_p, c_void_p)
def llama_log_callback(level, text, user_data):
    pass


llama_log_set(llama_log_callback, None)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Конфигурации моделей
TRANSFORMERS_MODEL = "IlyaGusev/saiga_mistral_7b_lora"
GGUF_MODEL_NAME = "IlyaGusev/saiga_mistral_7b_gguf"
GGUF_MODEL_FILE = str(Path(__file__).parent.parent /
                      "models" / "model-q8_0.gguf")

# Проверка перед загрузкой
if not Path(GGUF_MODEL_FILE).exists():
    raise FileNotFoundError(
        f"GGUF модель не найдена по пути: {GGUF_MODEL_FILE}")

# Системный промпт
DEFAULT_SYSTEM_PROMPT = (
    "Ты — AAA (Augmented Artificial Assistant), русскоязычный автоматический ассистент. "
    "Ты разговариваешь с людьми и помогаешь им. Отвечай вежливо."
)


class Conversation:
    def __init__(self):
        self.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        # </s> в конце каждого сообщения
        self.message_template = "{role}\n{content}</s>"
        self.response_template = "<s>bot\n"  # <s> только в начале ответа бота

    def get_prompt(self) -> str:
        prompt = ""
        for message in self.messages:
            # Для первого сообщения (system) добавляем <s> в начале
            if message == self.messages[0]:
                prompt += f"<s>{self.message_template.format(**message)}"
            else:
                prompt += f"<s>{self.message_template.format(**message)}"
        prompt += self.response_template
        return prompt.strip()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})


# ================== Обработчики моделей ==================

class GGUFHandler:
    def __init__(self):
        self.model = None
        self.use_gpu = self._check_gpu_support()
        self.load_model()

    def _check_gpu_support(self) -> bool:
        try:
            from llama_cpp.llama import llama_backend_init
            llama_backend_init(numa=False)
            return True
        except:
            return False

    def load_model(self):
        try:
            logger.info("Загрузка GGUF модели...")
            self.model = Llama(
                model_path=GGUF_MODEL_FILE,
                n_ctx=2048,
                n_gpu_layers=-1 if self.use_gpu else 0,
                n_threads=8 if not self.use_gpu else None
            )
            logger.info(
                f"GGUF модель загружена на {'GPU' if self.use_gpu else 'CPU'}")
        except Exception as e:
            logger.error(f"Ошибка загрузки GGUF модели: {e}")

    def generate(self, prompt: str) -> str:
        output = self.model(prompt, max_tokens=4096, stop=[
                            "</s>"], temperature=0.7, top_p=0.9)
        return output['choices'][0]['text'].strip()


# ================== Инициализация обработчиков ==================
conversation = Conversation()
gguf_handler = GGUFHandler()


# ================== Основные функции ==================
def generate_response(
    user_message: str,
    model_type: str = "gguf",
    use_conversation: bool = True
) -> str:
    """Генерация ответа с выбором модели"""
    if use_conversation:
        conversation.add_message("user", user_message)
        prompt = conversation.get_prompt()
    else:
        prompt = f"<s>system\n{DEFAULT_SYSTEM_PROMPT}\nuser\n{user_message}\nbot\n</s>"

    try:
        start_time = time.time()

        if model_type == "gguf":
            response = gguf_handler.generate(prompt)

        eval_time = time.time() - start_time
        logger.info(f"Генерация заняла {eval_time:.2f} сек")

        if use_conversation:
            conversation.add_message("bot", response)

        return response
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        return f"Ошибка: {str(e)}"

# TODO: Функция для генерации ответа с RAG


def generate_response_with_RAG(
    user_message: str,
    model_type: str = "gguf",
    threshold: float = 0.461,
    top_k: int = 3,
    debug_mode: bool = False
) -> str:
    """
    Генерация ответа с использованием RAG (Retrieval-Augmented Generation)

    Args:
        user_message: Входное сообщение пользователя
        model_type: Тип модели для генерации ("gguf")
        threshold: Порог косинусного сходства для релевантности документов
        top_k: Количество документов для поиска
        debug_mode: Режим отладки

    Returns:
        Ответ модели с учетом найденных документов или из собственных знаний
    """
    # 1. Поиск релевантных документов
    search_results = semantic_search(
        query=user_message,
        top_k=top_k,
        debug_mode=debug_mode
    )
    logger.info(
        f'семантический поиск нашёл {len(search_results)} ответов. начинаемм фильрацию через threshold...')
    if debug_mode:
        for i, hit in enumerate(search_results):
            dist = hit['distance']
            text = hit['entity']['text_content']
            print(f'{i+1}) [{ dist }] - {text} ')
    # 2. Фильтрация результатов по порогу
    relevant_docs = [
        hit for hit in search_results
        if hit['distance'] >= threshold
    ]
    logger.info(
        f'через пороговое значение в {threshold} прошло {len(relevant_docs)} документов...')

    # 3. Формирование контекста для LLM
    context = ""
    if relevant_docs:
        context = ""
        for i, hit in enumerate(relevant_docs):
            if hit['distance'] >= threshold:
                hit_text = hit['entity']['text_content']
                context += "\n\n" + \
                    f"Документ {i+1} Текст: \n{hit_text}"
            else:
                break
        if debug_mode:
            logger.info(f"Сформирован контекст: {context}")

    # 4. Формирование промпта в зависимости от наличия контекста
    if context:
        logger.info(f'Переходим в промпт с контекстом')
        prompt = (
            f"<s>system\n{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"Пользователь задал вопрос: {user_message}\n\n"
            f"Вот релевантная информация из документов:\n{context}\n\n"
            "Проверь эту информацию, чтобы дать точный ответ пользователю. В ней может и не быть нужнойинформации. Если в документах нет ответа, "
            "обязательно скажи, что не нашел информации в документах, но попробуй ответить из своих знаний."
            "</s><s>bot\n"
        )
    else:
        logger.info(f'Переходим в промпт без контекста:')
        prompt = (
            f"<s>system\n{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"Пользователь задал вопрос: {user_message}\n\n"
            "Релевантной информации в документах не найлено. Скажи об этом пользователю и ответь, используя свои знания. "
            "Если не знаешь ответа, так и скажи.</s><s>bot\n"
        )

    # 5. Генерация ответа
    try:
        logger.info('начало генерации ответа моделью...')
        start_time = time.time()
        if model_type == "gguf":
            if debug_mode:
                logger.info(f'Промпт: {prompt}')
            response = gguf_handler.generate(prompt)

        # Пост-обработка ответа
        response = response.strip()
        eval_time = time.time() - start_time
        logger.info(
            f"Генерация заняла {int(eval_time/60)} мин {eval_time%60:.2f} сек")
        if not response or response.lower() in ["не знаю", "не могу ответить"]:
            return "К сожалению, я не нашел подходящей информации ни в документах, ни в своих знаниях."

        return response

    except Exception as e:
        logger.error(f"Ошибка генерации с RAG: {e}")
        return "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже."
