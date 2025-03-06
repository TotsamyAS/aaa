import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# Импортируем функции из database.py
from modules.database import get_embeddings, search_documents

# Определяем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

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
    print("Начало загрузки модели...")
    config = PeftConfig.from_pretrained(MODEL_NAME)
    print("Конфигурация модели загружена.")

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_8bit=device == "cuda",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    print("Базовая модель загружена.")

    model = PeftModel.from_pretrained(
        model,
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    print("LoRA-адаптеры загружены.")

    model.eval()
    print("Модель переведена в режим eval.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    print("Токенизатор загружен.")

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    print("Конфигурация генерации загружена.")

    print("Модель успешно загружена.")
    model_ready = True
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")

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

# Функция для генерации ответа с RAG


def generate_response(user_message):
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
