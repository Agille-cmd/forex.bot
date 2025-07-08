import os
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

class BotConfig:
    """Класс конфигурации бота"""
    
    # Основные настройки
    TOKEN = os.getenv('7844378262:AAHmBSGea0znCCks3hsxpixCjEYeczwalsA')
    TWELVE_DATA_KEY = os.getenv('80caddf1ddcd4f1b983d21d31c8bc180')
    BOT_ID = None
    ALLOWED_USER_IDS = {7785586524, 7064593003}
    
    # Константы дизайна
    COLORS = {
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
    
    # Настройки индикаторов по умолчанию
    DEFAULT_SETTINGS = {
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
    
    # Группы валют
    CURRENCY_GROUPS = {
        'EUR': ['EUR/USD', 'EUR/GBP', 'EUR/JPY', 'EUR/AUD', 'EUR/CAD', 'EUR/CHF'],
        'USD': ['USD/JPY', 'USD/CAD', 'USD/CHF', 'AUD/USD', 'GBP/USD', 'NZD/USD'],
        'GBP': ['GBP/USD', 'GBP/JPY', 'GBP/AUD', 'GBP/CAD', 'GBP/CHF', 'GBP/NZD'],
        'AUD': ['AUD/USD', 'AUD/JPY', 'AUD/NZD', 'AUD/CAD', 'AUD/CHF'],
        'JPY': ['USD/JPY', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'CAD/JPY', 'CHF/JPY'],
        'CAD': ['USD/CAD', 'EUR/CAD', 'GBP/CAD', 'AUD/CAD', 'CAD/JPY', 'CAD/CHF'],
        'CHF': ['USD/CHF', 'EUR/CHF', 'GBP/CHF', 'AUD/CHF', 'CAD/CHF', 'CHF/JPY'],
        'BTC': ['BTC/USD', 'BTC/EUR', 'BTC/JPY', 'BTC/GBP']
    }
    
    # Таймфреймы
    TIMEFRAMES = ['5M', '15M', '1H', '4H']
    MULTI_TIMEFRAMES = ['15M', '1H', '4H']
    
    # Свечные паттерны
    CANDLE_PATTERNS = {
        'hammer': lambda o, h, l, c: (h-l > 3*(c-o)) and ((c-o) > 0) and ((o-l)/(0.001+h-l) > 0.6) and ((h-c)/(0.001+h-l) < 0.4),
        'inverse_hammer': lambda o, h, l, c: (h-l > 3*(o-c)) and ((o-c) > 0) and ((h-o)/(0.001+h-l) > 0.6) and ((c-l)/(0.001+h-l) < 0.4),
        'bullish_engulfing': lambda o1, h1, l1, c1, o2, h2, l2, c2: (c1 < o1) and (c2 > o2) and (c2 > o1) and (o2 < c1),
        'bearish_engulfing': lambda o1, h1, l1, c1, o2, h2, l2, c2: (c1 > o1) and (c2 < o2) and (c2 < o1) and (o2 > c1),
        'morning_star': lambda o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3: (c1 < o1) and (abs(c2-o2) < (h2-l2)*0.3) and (c3 > o3) and (c3 > ((o1 + c1)/2))
    }

# Создаем экземпляр конфигурации
config = BotConfig()
