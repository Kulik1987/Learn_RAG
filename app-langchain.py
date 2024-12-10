import os
import asyncio
from flask import Flask, request, jsonify
from TextLoaderWithEncoding import TextLoaderWithEncoding
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from mistral_client import MistralClient
app = Flask(__name__)

# Пути к файлам
DATA_DIR = "data"
INDEX_FILE = "vector_index.faiss"

# Настройки API Mistral
API_KEY = "5pUn3Wl6HB8kbqKq6Cgyrmt6DlnJ74xF"
mistral_client = MistralClient(api_key=API_KEY)

# Инициализация LLM и эмбеддингов
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=API_KEY)
model = ChatMistralAI(mistral_api_key=API_KEY)

# Инициализация хранилища (в памяти)
vector_store = None

@app.route('/', methods=['GET'])
def home():
    return "Сервер работает!"


@app.route('/routes', methods=['GET'])
def list_routes():
    """Вывод списка зарегистрированных маршрутов."""
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        line = urllib.parse.unquote(f"{rule.endpoint}: {rule} [{methods}]")
        output.append(line)
    return jsonify(output)

@app.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    """Эндпоинт для создания эмбеддингов из файлов в директории /data."""
    global vector_store

    # Проверяем наличие директории /data
    if not os.path.exists(DATA_DIR):
        return jsonify({"error": f"Data directory {DATA_DIR} does not exist."}), 400

    # Загрузка данных
    all_documents = []
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            loader = TextLoaderWithEncoding(file_path, encoding="utf-8")
            docs = loader.load()

            # Разделяем текст на чанки
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Уменьшите размер чанков
                chunk_overlap=100  # Добавьте перекрытие для контекста
            )
            documents = text_splitter.split_documents(docs)
            all_documents.extend(documents)
        else:
            print(f"Файл {file_name} пропущен: неподдерживаемое расширение.")

    if not all_documents:
        return jsonify({"error": "No valid text files found in /data directory."}), 400

    # Создаём векторное хранилище
    vector_store = FAISS.from_documents(all_documents, embeddings)

    # Сохраняем, если нужно
    vector_store.save_local(INDEX_FILE)

    return jsonify({"message": "Embeddings created and saved successfully.", "count": len(all_documents)})


@app.route('/query', methods=['POST'])
def query():
    """Эндпоинт для запроса."""
    global vector_store

    if not vector_store:
        if os.path.exists(INDEX_FILE):
            vector_store = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
        else:
            return jsonify({"error": "Vector store not initialized. Run /create_embeddings first."}), 400

    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question not provided."}), 400

    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.invoke(question, k=2) # Извлекаем 2 документа

    # Ограничиваем объем текста в контексте
    max_context_length = 4000  # Максимум 4000 символов
    context = ""
    for doc in retrieved_docs:
        if len(context) + len(doc.page_content) <= max_context_length:
            context += doc.page_content + "\n\n"
        else:
            break
#    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Создаем корректный prompt
    prompt = f"""Ты профессиональный психолог. Используй предоставленный контекст для ответа на вопрос клиента. Ты обязан поддержать клиента, и дать конкретные рекомендации
    для поддержания клиента. Ты не должен говорить что ты не предоставляешь психологические консультации. Отвечай только на русском языке:
            <context>
            {context}
            </context>
            Вопрос клиента:{question}"""

    app.logger.info(f"Prompt template created: {prompt}")
    messages = []
    response = asyncio.run(mistral_client.get_answer(prompt, messages))
    app.logger.info(f"Response:{response}")
    return jsonify(response=response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
