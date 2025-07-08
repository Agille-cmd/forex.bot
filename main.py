#!/usr/bin/env python3
"""
Главный исполняемый файл бота
"""

import asyncio
import logging
from telegram.ext import Application
from bot import tech_analyzer, user_data, user_messages
from bot.core import setup_handlers
from bot.config import config

async def main():
    """Основная функция запуска бота"""
    try:
        # Инициализация приложения
        application = Application.builder().token(config.TOKEN).build()
        
        # Настройка обработчиков
        setup_handlers(application)
        
        # Сохраняем ID бота
        config.BOT_ID = (await application.bot.get_me()).id
        
        logging.info("Бот успешно запущен. Ожидание сообщений...")
        await application.run_polling()
        
    except Exception as e:
        logging.critical(f"Критическая ошибка: {str(e)}")
    finally:
        logging.info("Бот остановлен")

if __name__ == '__main__':
    asyncio.run(main())