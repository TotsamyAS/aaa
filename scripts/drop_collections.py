from pymilvus import connections, Collection, MilvusClient

URI = "http://localhost:19530"
TOKEN = "root:Milvus"
# Подключение к Milvus
connections.connect("default", host="localhost", port="19530")

client = MilvusClient(uri=URI, token=TOKEN)

client.drop_collection("document_db_collection")
client.drop_collection("document_name_collection")
print('Collections dropped')
