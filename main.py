from bot.config import config  # Импортируем объект config
from bot import tech_analyzer, user_data, user_messages
from bot.core import setup_handlers
from telegram.ext import Updater
import logging

def main():
    try:
        updater = Updater(config.TOKEN, use_context=True)  # Используем config.TOKEN
        setup_handlers(updater.dispatcher)
        
        logging.info("Бот запущен. Ожидание сообщений...")
        updater.start_polling()
        updater.idle()
    except Exception as e:
        logging.critical(f"Ошибка: {str(e)}")

if __name__ == '__main__':
    main()
