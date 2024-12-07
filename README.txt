Для запуска выполнить команду в командной строке
python .\app-langchain.py


!!!!!! скорее всего потребуется установить много библиотек для запуска - недостающие добавить в файл requerments.txt
загрузку библиотек можно делать через вызов этого файла

После этого доступны 2 ендпоинта(можно вызывтаь через postman или через любой другой инстурмент для вызова http методов)



Вопрос - поиск происходит только по базе знаний, без участия ИИ
POST http://127.0.0.1:5000/query
Body - выбрать row тип json
{
    "question": "Опиши игру алкоголик?"
}

ответ
{
    "answer": "The game \"Alcoholic\" is a social interaction game that involves five main roles: the Alcoholic (referred to as \"White\"), the Persecutor (usually played by the spouse), the Rescuer (often a person of the same gender, such as a doctor interested in alcoholism), the Fool (who provides alcohol to the Alcoholic on credit or loan), and the Mediator (who understands the Alcoholic's condition and is often the main person in the Alcoholic's life). The game's primary objective for the Alcoholic is not the pleasure of drinking but achieving a hangover, which is perceived as a psychological torture. The game can have two to five participants, and some roles may be combined. The game's dynamics involve social transactions related to alcoholism.",
    "question": "Опиши игру алкоголик?"
}

POST http://127.0.0.1:5000/create_embeddings
тело запроса пустое

response: статус 200 означает что все ок; 500 - ошибка
{
    "count": 3906,
    "message": "Embeddings created and saved successfully."
}