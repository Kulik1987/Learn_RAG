@echo off

:: Создание виртуального окружения
python -m venv mistral-env

:: Активация виртуального окружения
call mistral-env\Scripts\activate

:: Установка зависимостей
pip install --upgrade pip
pip install transformers datasets bitsandbytes accelerate peft

echo Окружение настроено. Активируйте его с помощью "mistral-env\Scripts\activate".