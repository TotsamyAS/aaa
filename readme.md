# AAA (Augmented Artificial Assistant)

Это дипомный проект, по созданию интерактивного чат-бота на основе модели Saiga Mistral 7B и технологии RAG (Retrieval Augmented Generation).

# Запуск приложения

## Режим разработки

1. Запускаем виртуальную срежду:

```{bash}
.venv\Scripts\activate
```

2. Запуск приложения:

```{bash}
python app.py
```

Образ запустится на хосте `http://localhost:5000`

3. Останавливаем приложение

Для этого служит сочетание клавиш `Ctrl + C`

4. Сбросить БД

```{bash}
python ./scripts/drop_collections.py
```

# Запус базы данных

1. Включить

- milvus-standalone: порт 19530
- embed etcd: порт 2370

```
./milvus_standalone.bat start
```

2. Остановить

```
./milvus_standalone.bat stop
```

3. Удалить контейнер

```
./milvus_standalone.bat delete
```
