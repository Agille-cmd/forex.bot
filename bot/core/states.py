import time
from typing import Dict, Any, List

# Глобальные переменные состояния
user_data: Dict[int, Dict[str, Any]] = {}
user_messages: Dict[int, List[int]] = {}

def cleanup_old_users():
    """Очистка данных неактивных пользователей."""
    now = time.time()
    for user_id in list(user_data.keys()):
        if now - user_data[user_id].get('last_activity', 0) > 86400:  # 24 часа
            del user_data[user_id]
    for user_id in list(user_messages.keys()):
        if user_id not in user_data:
            del user_messages[user_id]