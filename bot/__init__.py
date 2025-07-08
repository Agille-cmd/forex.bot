"""
Инициализация основного модуля бота

Этот файл объединяет все компоненты бота и обеспечивает их корректную работу
"""

import logging
from typing import Dict, Any, List

# Инициализация глобальных переменных состояния
user_data: Dict[int, Dict[str, Any]] = {}
user_messages: Dict[int, List[int]] = {}

# Настройка логгера для всего модуля
logger = logging.getLogger(__name__)

def init_bot():
    """Инициализация всех компонентов бота"""
    from bot.core import setup_handlers
    from bot.utils.helpers import setup_logging
    from bot.services import TechnicalAnalyzer
    
    # Настройка логирования
    setup_logging()
    logger.info("Инициализация бота...")
    
    # Создание экземпляра анализатора
    tech_analyzer = TechnicalAnalyzer()
    
    logger.info("Все компоненты бота успешно инициализированы")
    return tech_analyzer

# Автоматическая инициализация при импорте
tech_analyzer = init_bot()

__all__ = [
    'tech_analyzer',
    'user_data',
    'user_messages'
]