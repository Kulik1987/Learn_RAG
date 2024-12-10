from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained("./llama_fine_tuned")
tokenizer = AutoTokenizer.from_pretrained("./llama_fine_tuned")

# Пример ввода
prompt = "Translate the following text to French:\n\nHello, how are you?"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Результат генерации:", result)