from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType

# Загрузка токенизированных данных
tokenized_dataset = load_from_disk("./tokenized_dataset")

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained("./llama_model")

# Настройка LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Количество рангов для адаптации
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="llama_fine_tuned",
    per_device_train_batch_size=1,  # Маленький размер батча для CPU
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    logging_dir="logs",
    learning_rate=5e-5,
    save_steps=500,
    save_total_limit=2,
    fp16=False,  # Отключено для CPU
    optim="adamw_torch"
)

# Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Сохранение модели
model.save_pretrained("./llama_fine_tuned")
print("Fine-tuning завершен. Модель сохранена в './llama_fine_tuned'.")
