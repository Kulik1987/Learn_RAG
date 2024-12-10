from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Вставьте сюда ваш токен
api_token = "your_token"
login(api_token)

print("Успешно авторизован.")
# Имя модели
model_name = "andrijdavid/Llama3-1B-Base"

# Скачиваем модель и токенизатор
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Сохраняем модель локально
model.save_pretrained("./llama_model")
tokenizer.save_pretrained("./llama_model")

print("Модель успешно загружена и сохранена в './llama_model'.")
