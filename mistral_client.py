import logging
from mistralai.client import MistralClient as OriginalMistralClient
from mistralai.models.chat_completion import ChatMessage

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MistralClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OriginalMistralClient(api_key=api_key)
        self.chunks = None
        self.index = None

    async def get_answer(self, client_message: str, messages: list):
        """ Формирование нового сообщения от клиента, добавление его в список сообщений.
            *пока не реализовано: Поиск наиболее релевантных кусков текста и генерация ответа на основе них."""
        message_with_promt = "Вопрос клиента:" + (client_message or "Неизвестный вопрос")
        message = ChatMessage(role="user", content=message_with_promt)
        messages.append(message)
        return await self.run_mistral(messages)

    async def run_mistral(self, messages: list):
        """Отправка сообщения в модель Mistral и получение ответа."""
        try:
            model = "mistral-large-latest"
            chat_response = self.client.chat(model=model, messages=messages, temperature=0.6, top_p=0.6)
            return chat_response.choices[0].message.content
        except Exception as e:
            error_message = f"Ошибка при взаимодействии с моделью Mistral: {e}"
            logger.error(error_message)
            await self.post_alert(error_message)
            return "Произошла ошибка при обработке вашего запроса. Пожалуйста, повторите попытку позже."
