from flask import Flask, request, jsonify, render_template, redirect, url_for
from pymilvus import connections, Collection
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from database import upload_pdf_to_db, delete_pdf_from_db, initialize_collections, get_uploaded_documents, drop_collections

import sys
sys.path.insert(0, "modules")


app = Flask(__name__, template_folder="templates")

# Инициализация коллекций при запуске приложения
initialize_collections()

# Главная страница для загрузки PDF


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

# Страница для удаления PDF


@app.route("/delete", methods=["GET"])
def delete_page():
    return render_template("delete.html")

# Эндпоинт для загрузки PDF


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400

    # Используем имя файла как doc_name
    doc_name = file.filename

    # Сохраняем файл временно
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, doc_name)
    file.save(file_path)

    # Загружаем PDF в базу данных
    upload_pdf_to_db(file_path, doc_name)

    # Удаляем временный файл
    os.remove(file_path)

    # Перенаправляем на главную страницу с сообщением об успехе
    return redirect(url_for("index", message=f"Документ '{doc_name}' успешно загружен"))

# Эндпоинт для удаления PDF


@app.route("/delete", methods=["POST"])
def delete_pdf():
    doc_name = request.form.get("doc_name")
    if not doc_name:
        return jsonify({"error": "Название документа не указано"}), 400

    # Удаляем документ из базы данных
    delete_pdf_from_db(doc_name)

    # Перенаправляем на страницу удаления с сообщением об успехе
    return redirect(url_for("delete_page", message=f"Документ '{doc_name}' успешно удалён"))


@app.route("/docs", methods=["GET"])
def show_docs():
    documents = get_uploaded_documents()
    return render_template("upload.html", documents=documents)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
