from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path("F:/mistral_models/22B-Instruct-Small")
mistral_models_path.mkdir(parents=True, exist_ok=True)

# Скачиваем необходимые файлы
snapshot_download(
    repo_id="mistralai/Mistral-Small-Instruct-2409",
    allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
    local_dir=mistral_models_path
)
