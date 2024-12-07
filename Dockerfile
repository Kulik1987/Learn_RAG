# Используем официальный образ Python в качестве базового
FROM python:3.11-slim

# Устанавливаем зависимости для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы в контейнер
COPY .. .

# Указываем рабочую директорию
WORKDIR /

# Запускаем Flask приложение
CMD ["python", "app.py"]
