import os
import faiss
import numpy as np
import pickle
from flask import Flask, request, jsonify
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

app = Flask(__name__)

# Настройки
API_KEY = "5pUn3Wl6HB8kbqKq6Cgyrmt6DlnJ74xF"
INDEX_FILE = "vector_index7.faiss"
CHUNKS_FILE = "chunks7.pkl"
EMBEDDING_CACHE_FILE = "embedding_cache.pkl"

# Глобальный массив для хранения сообщений
messages = []

# Инициализация Mistral Client
client = MistralClient(api_key=API_KEY)

# Загрузка индекса и чанков
index = None
chunks = None

# Кэш для эмбеддингов
embedding_cache = {}

# Загрузка кэша, если файл существует
if os.path.exists(EMBEDDING_CACHE_FILE):
    with open(EMBEDDING_CACHE_FILE, "rb") as f:
        embedding_cache = pickle.load(f)

# Загрузка индекса и чанков из файлов
if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        data = pickle.load(f)

        # Проверяем, является ли data списком или словарем
        if isinstance(data, list):
            chunks = data
        elif isinstance(data, dict) and 'chunks' in data:
            chunks = data['chunks']
        else:
            raise ValueError("Файл chunks6.pkl имеет неподдерживаемый формат.")

def get_text_embedding(input_text):
    """Получение эмбеддинга с проверкой кэша."""
    if input_text in embedding_cache:
        return embedding_cache[input_text]

    embeddings_batch_response = client.embeddings(
        model="mistral-embed",
        input=input_text
    )
    embedding = embeddings_batch_response.data[0].embedding

    # Сохраняем эмбеддинг в кэше
    embedding_cache[input_text] = embedding

    # Обновляем файл кэша
    with open(EMBEDDING_CACHE_FILE, "wb") as f:
        pickle.dump(embedding_cache, f)

    return embedding

@app.route('/query', methods=['POST'])
def query():
    global index, chunks, messages
    data = request.json
    question = data.get('question')

    if not question or not index or not chunks:
        return jsonify({"error": "Invalid input or database not initialized"}), 400

    # Генерация эмбеддинга для вопроса с использованием кэша
    question_embedding = np.array([get_text_embedding(question)])
    D, I = index.search(question_embedding, k=2)
    retrieved_chunk = [chunks[i] for i in I[0]]

    prompt = f"""
    Контекстная информация приведена ниже.
    ---------------------
    {retrieved_chunk}
    ---------------------
    # Персонаж
    Вы - умелый психолог. Вы способны поддержать человека в его проблеме, для формирования ответа используй контекстную информацию.
    Если информации недостаточно, задавай уточняющие вопросы. 
    
    ## Навыки
    ### Навык 1: Понимание проблемы
    - Внимательно слушайте пользователя и идентифицируйте проблему, с которой он столкнулся.
    
    ### Навык 2: Поддержка и помощь
    - Проявите чуткость и внимательность, привлекайте свой профессиональный опыт и знания, чтобы поддержать пользователя и помочь ему преодолеть проблему.
    
    ### Навык 3: Ответ на запрос
    - На основе полученной информации ответьте на запрос пользователя 
    
    ## Ограничения:
    - Если необходимы уточняющие вопросы, в 1 сообщении задавай только 1 вопрос
    - Отвечайте только на русском языке.
    - Будьте внимательным слушателем, ваши ответы должны быть поддерживающими.
    - Не упоминайте ссылки на документы или статьи, пишите свои ответы так, как будто это ваши мысли.
    - Ваши ответы должны быть направлены на поддержку человека в его проблеме.
    Query: {question}
    Answer:
    """

    def run_mistral(user_message, model="mistral-small-latest"):
        messages.append({"role": "user", "content": user_message})
        chat_response = client.chat(
            model=model,
            messages=messages
        )
        response_message = chat_response.choices[0].message.content
        messages.append({"role": "system", "content": response_message})
        return response_message

    answer = run_mistral(prompt)
    return jsonify({"question": question, "answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)