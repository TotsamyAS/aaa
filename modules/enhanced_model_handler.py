import torch

import logging
import time
from pathlib import Path
import os

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from llama_cpp import Llama
from vllm import LLM, SamplingParams


# Импортируем функции из database.py
from modules.database import get_embeddings, search_documents


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
    "Ты разговариваешь с людьми и помогаешь им."
)


class Conversation:
    def __init__(self):
        self.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        self.message_template = "<s>{role}\n{content}</s>"
        self.response_template = "<s>bot\n"

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_prompt(self) -> str:
        prompt = ""
        for message in self.messages:
            prompt += self.message_template.format(**message)
        prompt += self.response_template
        return prompt.strip()


# ================== Обработчики моделей ==================


class TransformersHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.offload_folder = "./models/offload"  # Папка для оффлоуда
        # Создаем папку если нет
        os.makedirs(self.offload_folder, exist_ok=True)
        self.load_model()

    def load_model(self):
        try:
            logger.info(
                f"Загрузка модели библиотеки Transformers на {self.device}...")
            # Загружаем конфигурацию
            config = PeftConfig.from_pretrained(TRANSFORMERS_MODEL)

            # Загружаем базовую модель с указанием папки для оффлоуда
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True,
                    offload_folder=self.offload_folder
                )
            # Вариант 2: Для CPU или GPU с малым объемом памяти
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )

            self.model = PeftModel.from_pretrained(
                self.model,
                TRANSFORMERS_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                TRANSFORMERS_MODEL,
                use_fast=False
            )

            self.model.eval()
            logger.info(
                f"Transformers модель загружена на {self.device.upper()}")
        except Exception as e:
            logger.error(f"Ошибка загрузки Transformers модели: {e}")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        output_ids = self.model.generate(**inputs)[0]
        return self.tokenizer.decode(output_ids[len(inputs["input_ids"][0]):], skip_special_tokens=True)


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
        output = self.model(prompt, max_tokens=256, stop=["</s>"])
        return output['choices'][0]['text'].strip()


class VLLMHandler:
    def __init__(self):
        self.llm = None
        self.use_gpu = torch.cuda.is_available()
        self.load_model()

    def load_model(self):
        try:
            logger.info("Загрузка vLLM модели...")
            self.llm = LLM(
                model=TRANSFORMERS_MODEL,
                tensor_parallel_size=1 if self.use_gpu else None,
                dtype="auto"
            )
            logger.info(
                f"vLLM модель загружена на {'GPU' if self.use_gpu else 'CPU'}")
        except Exception as e:
            logger.error(f"Ошибка загрузки vLLM модели: {e}")

    def generate(self, prompt: str) -> str:
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=256, stop=["</s>"])
        output = self.llm.generate(prompt, sampling_params)
        return output[0].outputs[0].text.strip()


# ================== Инициализация обработчиков ==================
conversation = Conversation()
transformers_handler = TransformersHandler()
gguf_handler = GGUFHandler()
vllm_handler = VLLMHandler()


# ================== Основные функции ==================
def generate_response(
    user_message: str,
    model_type: str = "transformers",  # "transformers", "gguf", "vllm"
    use_conversation: bool = True
) -> str:
    """Генерация ответа с выбором модели"""
    if use_conversation:
        conversation.add_message("user", user_message)
        prompt = conversation.get_prompt()
    else:
        prompt = f"<s>system\n{DEFAULT_SYSTEM_PROMPT}</s><s>user\n{user_message}</s><s>bot\n"

    try:
        start_time = time.time()
        if model_type == "transformers":  # transformers по умолчанию
            response = transformers_handler.generate(prompt)
        elif model_type == "gguf":
            response = gguf_handler.generate(prompt)
        elif model_type == "vllm":
            response = vllm_handler.generate(prompt)

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
    model_type: str = "transformers"
) -> str:
    pass
