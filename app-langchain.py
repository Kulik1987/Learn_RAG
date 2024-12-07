import os
from flask import Flask, request, jsonify
from TextLoaderWithEncoding import TextLoaderWithEncoding
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

# Пути к файлам
DATA_DIR = "data"
INDEX_FILE = "vector_index.faiss"

# Настройки API Mistral
API_KEY = "5pUn3Wl6HB8kbqKq6Cgyrmt6DlnJ74xF"

# Инициализация LLM и эмбеддингов
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=API_KEY)
model = ChatMistralAI(mistral_api_key=API_KEY)

# Инициализация хранилища (в памяти)
vector_store = None


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
            text_splitter = RecursiveCharacterTextSplitter()
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
        # Загружаем хранилище, если оно существует
        if os.path.exists(INDEX_FILE):
            vector_store = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
        else:
            return jsonify({"error": "Vector store not initialized. Run /create_embeddings first."}), 400

    # Получаем вопрос из запроса
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question not provided."}), 400

    # Настройка запроса
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Выполнение запроса
    response = retrieval_chain.invoke({"input": question})
    return jsonify({"question": question, "answer": response["answer"]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
