import os
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

class BotConfig:
    def __init__(self):
        # Основные настройки
        self.TOKEN = os.getenv('TELEGRAM_TOKEN')
        self.TWELVE_DATA_KEY = os.getenv('TWELVE_DATA_API_KEY')
        self.BOT_ID = None
        self.ALLOWED_USER_IDS = {7785586524, 7064593003}
        
        # Константы дизайна
        self.COLORS = {
            'up': '#2ecc71',
            'down': '#e74c3c',
            'wick_up': '#27ae60',
            'wick_down': '#c0392b',
            'ema_fast': '#3498db',
            'ema_slow': '#9b59b6',
            'supertrend': '#f39c12',
            'grid': '#ededed',
            'text': '#2c3e50',
            'background': 'white'
        }
        
        # Настройки индикаторов
        self.DEFAULT_SETTINGS = {
            'RSI_PERIOD': 14,
            'MACD_FAST': 12,
            'MACD_SLOW': 26,
            'MACD_SIGNAL': 9,
            'BB_PERIOD': 20,
            'BB_STDDEV': 2,
            'STOCH_K': 14,
            'STOCH_D': 3,
            'ADX_PERIOD': 14,
            'SUPERTREND_PERIOD': 10,
            'SUPERTREND_MULTIPLIER': 3.0,
            'ICHIMOKU_TENKAN': 9,
            'ICHIMOKU_KIJUN': 26,
            'ICHIMOKU_SENKOU': 52,
            'EMA_FAST': 50,
            'EMA_SLOW': 200
        }
        
        # Группы валют и таймфреймы
        self.CURRENCY_GROUPS = {
            'EUR': ['EUR/USD', 'EUR/GBP', 'EUR/JPY', 'EUR/AUD', 'EUR/CAD', 'EUR/CHF'],
            # ... остальные валютные группы
        }
        
        self.TIMEFRAMES = ['5M', '15M', '1H', '4H']
        self.MULTI_TIMEFRAMES = ['15M', '1H', '4H']
        
        # Свечные паттерны
        self.CANDLE_PATTERNS = {
            # ... ваши паттерны
        }

# Создаем и экспортируем экземпляр конфигурации
config = BotConfig()
