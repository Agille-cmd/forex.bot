#!/usr/bin/env python3
from telegram.ext import Updater
from bot import tech_analyzer, user_data, user_messages
from bot.core import setup_handlers
from bot.config import config
import logging

def main():
    """Основная функция запуска бота"""
    try:
        # Инициализация приложения
        updater = Updater(config.TOKEN, use_context=True)
        
        # Настройка обработчиков
        setup_handlers(updater.dispatcher)
        
        # Сохраняем ID бота
        config.BOT_ID = updater.bot.id
        
        logging.info("Бот успешно запущен. Ожидание сообщений...")
        updater.start_polling()
        updater.idle()
        
    except Exception as e:
        logging.critical(f"Критическая ошибка: {str(e)}")

if __name__ == '__main__':
    main()
