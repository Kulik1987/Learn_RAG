from datasets import load_dataset
from transformers import AutoTokenizer

# Загрузка данных
data_path = "/fine_tuning/llama_3_1_1B/test-data-train.json"  # Путь к данным
dataset = load_dataset("json", data_files=data_path, split="train")

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained("./llama_model")

# Добавляем pad_token, если он отсутствует
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Токенизация данных
def tokenize_function(examples):
    instructions = examples["instruction"]  # Список инструкций
    inputs = examples["input"]  # Список входных данных
    outputs = examples["output"]  # Список выходных данных

    # Генерация запросов (prompts)
    prompts = [
        str(instruction) + "\n\n" + str(input_text)
        for instruction, input_text in zip(instructions, inputs)
    ]

    # Токенизация
    return tokenizer(
        prompts,
        text_target=outputs,
        truncation=True,
        padding="max_length",  # Выравнивание длины
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Сохраняем токенизированные данные
tokenized_dataset.save_to_disk("./tokenized_dataset")
print("Данные успешно токенизированы и сохранены в './tokenized_dataset'.")
