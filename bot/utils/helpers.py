import logging
import time
from typing import Dict, Any, Optional, List
from telegram import InlineKeyboardMarkup
from telegram.ext import CallbackContext
from telegram.error import TelegramError
from bot.config import ALLOWED_USER_IDS, DEFAULT_SETTINGS

def setup_logging():
    """Настройка логирования для всего приложения."""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("bot.log", mode='a'),
            logging.StreamHandler()
        ]
    )

def check_access(user_id: int) -> bool:
    """Проверка, имеет ли пользователь доступ к боту."""
    return user_id in ALLOWED_USER_IDS

def ensure_user_data(user_id: int) -> Dict[str, Any]:
    """Создает структуру данных пользователя, если она не существует."""
    if user_id not in user_data:
        user_data[user_id] = {
            'settings': DEFAULT_SETTINGS.copy(),
            'last_activity': time.time(),
            'step': 'main_menu',
            'optimized_params': {}
        }
    user_data[user_id]['last_activity'] = time.time()
    return user_data[user_id]

def safe_answer_query(query, text: str = None, show_alert: bool = False) -> bool:
    """Безопасная обработка ответа на callback_query."""
    try:
        query.answer(text=text, show_alert=show_alert)
        return True
    except Exception as e:
        logging.warning(f"Ошибка при answer_query: {str(e)}")
        return False

def safe_edit_message(
    context: CallbackContext, 
    chat_id: int, 
    message_id: int, 
    text: str, 
    reply_markup: InlineKeyboardMarkup = None
) -> bool:
    """Безопасное редактирование сообщения."""
    try:
        if not text.strip():
            return False
            
        context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        return True
    except TelegramError as e:
        if "Message is not modified" not in str(e):
            logging.warning(f"Ошибка при edit_message: {str(e)}")
        return False
    except Exception as e:
        logging.warning(f"Ошибка при edit_message: {str(e)}")
        return False

def safe_delete_message(context: CallbackContext, chat_id: int, message_id: int) -> bool:
    """Безопасное удаление сообщения."""
    try:
        context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        return True
    except Exception as e:
        logging.warning(f"Ошибка при delete_message: {str(e)}")
        return False

def safe_send_message(
    context: CallbackContext,
    chat_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup = None
) -> Optional[int]:
    """Безопасная отправка сообщения."""
    try:
        msg = context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        return msg.message_id
    except Exception as e:
        logging.error(f"Ошибка при send_message: {str(e)}")
        return None

def add_message_to_cleanup(user_id: int, message_id: int):
    """Добавление сообщения в список для последующей очистки."""
    if user_id not in user_messages:
        user_messages[user_id] = []
    if message_id not in user_messages[user_id]:
        user_messages[user_id].append(message_id)

def cleanup_messages(context: CallbackContext, chat_id: int, user_id: int):
    """Очистка всех сообщений пользователя."""
    if user_id in user_messages:
        for msg_id in user_messages[user_id]:
            safe_delete_message(context, chat_id, msg_id)
        user_messages[user_id] = []