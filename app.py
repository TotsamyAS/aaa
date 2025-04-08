from flask import Flask, request, jsonify, render_template, redirect, url_for
from pymilvus import connections, Collection
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import os


from modules.database import upload_pdf_to_db, delete_pdf_from_db, initialize_collections, get_uploaded_documents, drop_collections
from modules.model_handler import generate_response, generate_response_with_RAG


app = Flask(__name__, template_folder="templates")

# Инициализация коллекций при запуске приложения
initialize_collections()


# Главная страница (документы)
@app.route("/")
def documents():
    # Получаем список документов из базы данных
    documents = get_uploaded_documents()
    return render_template("documents.html", documents=documents)

# Страница чата


@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400

    # Используем имя файла как doc_name
    doc_name = file.filename

    # Генерируем дату в формате ДД.ММ.ГГГГ
    date_added = datetime.now().strftime("%d.%m.%Y")

    # Сохраняем файл временно
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, doc_name)
    file.save(file_path)

    # Загружаем PDF в базу данных
    upload_pdf_to_db(file_path, doc_name, date_added)

    # Удаляем временный файл
    os.remove(file_path)

    # Перенаправляем на главную страницу с сообщением об успехе
    print(f"Документ '{doc_name}' успешно загружен")
    return redirect(url_for("documents", documents=get_uploaded_documents()))

# Эндпоинт для удаления PDF


@app.route("/delete", methods=["DELETE"])
def delete_pdf():
    # Получаем doc_name из query-параметров
    doc_name = request.args.get("doc_name")
    # Возвращаем успешный ответ
    if not doc_name:
        return jsonify({"error": "Название документа не указано"}), 400

    # Удаляем документ из базы данных
    delete_pdf_from_db(doc_name)

    # Возвращаем успешный ответ
    return jsonify({"message": f"Документ '{doc_name}' успешно удалён"}), 200


@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.get_json()
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "Сообщение не указано"}), 400

    # Генерируем ответ с помощью модели
    bot_response = generate_response_with_RAG(user_message)

    # Возвращаем ответ
    return jsonify({"message": bot_response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
