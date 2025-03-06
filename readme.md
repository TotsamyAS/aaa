# AAA (Augmented Artificial Assistant)

Это дипомный проект, по созданию интерактивного чат-бота на основе модели Saiga Mistral 7B и технологии RAG (Retrieval Augmented Generation).

# Установка приложения

1. Включить бд Milvus

- milvus-standalone: порт 19530
- embed etcd: порт 2370

```
./milvus_standalone.bat start
```

2. Создаём виртуальную среду. Рекоммендуется делать её на python 3.11

```
python -m venv .venv
```

3. Активируем виртуальную среду:

```
.venv\Scripts\activate
```

4. Устанавливаем пакеты
   (!!!) Убедитесь, что вы при этом находитесь внутри среды (!!!)
   Также предупреждаем, что установленные пакеты и модели займут пространство в `20 GB`, `13` из которых будут на диске `С`.

```
pip install -r requirements.txt
```

# Режим разработки

1. Останавливаем приложение

Для этого служит сочетание клавиш `Ctrl + C`

2. Сбросить БД (опц.)

```{bash}
python ./scripts/drop_collections.py
```

# Запуск и остановка базы данных

1. Включить

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
