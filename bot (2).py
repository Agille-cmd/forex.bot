import logging
import os
import io
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ChatAction, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters, CallbackContext

matplotlib.use('Agg')

# ================== КОНСТАНТЫ ДИЗАЙНА ==================
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

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TOKEN ='7844378262:AAHmBSGea0znCCks3hsxpixCjEYeczwalsA'
TWELVE_DATA_KEY ='80caddf1ddcd4f1b983d21d31c8bc180'

if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN не найден в .env файле")
if not TWELVE_DATA_KEY:
    raise ValueError("TWELVE_DATA_API_KEY не найден в .env файле")

# Конфигурация
BOT_ID = None
ALLOWED_USER_IDS = {7785586524, 7064593003}

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

user_data: Dict[int, Dict[str, Any]] = {}
user_messages: Dict[int, List[int]] = {}

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================
def check_access(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS

def ensure_user_data(user_id: int) -> Dict[str, Any]:
    if user_id not in user_data:
        user_data[user_id] = {
            'settings': DEFAULT_SETTINGS.copy(),
            'last_activity': time.time(),
            'step': 'main_menu',
            'optimized_params': {}
        }
    return user_data[user_id]

def get_user_settings(user_id: int) -> Dict[str, Any]:
    return ensure_user_data(user_id)['settings']

def safe_answer_query(query, text: str = None, show_alert: bool = False) -> bool:
    try:
        query.answer(text=text, show_alert=show_alert)
        return True
    except Exception as e:
        logger.warning(f"Ошибка при answer_query: {str(e)}")
        return False

def safe_edit_message(context: CallbackContext, chat_id: int, message_id: int, text: str, reply_markup=None) -> bool:
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
    except Exception as e:
        if "Message is not modified" not in str(e):
            logger.warning(f"Ошибка при edit_message: {str(e)}")
        return False

def safe_delete_message(context: CallbackContext, chat_id: int, message_id: int) -> bool:
    try:
        context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        return True
    except Exception as e:
        logger.warning(f"Ошибка при delete_message: {str(e)}")
        return False

def safe_send_message(context: CallbackContext, chat_id: int, text: str, reply_markup=None) -> Optional[int]:
    try:
        msg = context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        return msg.message_id
    except Exception as e:
        logger.error(f"Ошибка при send_message: {str(e)}")
        return None

def create_menu_buttons(buttons_data: list, back_button: bool = True) -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(text, callback_data=callback)] for text, callback in buttons_data]
    if back_button:
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data='back')])
    return InlineKeyboardMarkup(keyboard)

def cleanup_messages(context: CallbackContext, chat_id: int, user_id: int):
    if user_id in user_messages:
        for msg_id in user_messages[user_id]:
            safe_delete_message(context, chat_id, msg_id)
        user_messages[user_id] = []

def add_message_to_cleanup(user_id: int, message_id: int):
    if user_id not in user_messages:
        user_messages[user_id] = []
    user_messages[user_id].append(message_id)

def cleanup_and_send(context: CallbackContext, chat_id: int, user_id: int, text: str, buttons_data: list):
    cleanup_messages(context, chat_id, user_id)
    msg_id = safe_send_message(
        context,
        chat_id,
        text,
        reply_markup=create_menu_buttons(buttons_data)
    )
    if msg_id:
        add_message_to_cleanup(user_id, msg_id)

# ================== ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ ==================
def get_ohlc_data(symbol: str = "EUR/USD", timeframe: str = "1H") -> pd.DataFrame:
    try:
        interval_map = {'5M': '5min', '15M': '15min', '1H': '1h', '4H': '4h'}
        params = {
            'symbol': symbol,
            'interval': interval_map.get(timeframe, '1h'),
            'apikey': '80caddf1ddcd4f1b983d21d31c8bc180',
            'outputsize': 500,
            'format': 'JSON'
        }
        
        logger.info(f"Запрос данных для {symbol} ({timeframe})")
        response = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok' or 'values' not in data:
            raise ValueError(data.get('message', 'Неверный формат данных от API'))
            
        df = pd.DataFrame(data['values'])
        df = df.rename(columns={
            'datetime': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        numeric_cols = ['open', 'high', 'low', 'close']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        if df.empty:
            raise ValueError("Получены пустые данные")
            
        logger.info(f"Получены данные для {symbol} ({timeframe}), последняя точка: {df.index[-1]}")
        return df[['open', 'high', 'low', 'close']]
        
    except Exception as e:
        logger.error(f"Ошибка получения данных: {str(e)}\n{traceback.format_exc()}")
        return generate_test_data(symbol, timeframe)

def generate_test_data(symbol: str, timeframe: str) -> pd.DataFrame:
    logger.warning("Генерация тестовых данных")
    now = datetime.now()
    np.random.seed(int(now.timestamp()))
    
    periods_map = {'5M': 288, '15M': 96, '1H': 100, '4H': 50}
    base_price = 1.0 if 'USD' in symbol else 100.0
    prices = base_price + np.random.normal(0, 0.02, periods_map.get(timeframe, 100)).cumsum()
    dates = pd.date_range(end=now, periods=periods_map.get(timeframe, 100), freq=timeframe)
    
    return pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0.001, 0.002, periods_map.get(timeframe, 100))),
        'low': prices - np.abs(np.random.normal(0.001, 0.002, periods_map.get(timeframe, 100))),
        'close': prices + np.random.normal(0, 0.001, periods_map.get(timeframe, 100))
    }).set_index('date')

def find_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> List[Tuple[float, str]]:
    """Ищет уровни Фибоначчи по последнему трендовому движению"""
    recent = df[-lookback:]
    high = recent['high'].max()
    low = recent['low'].min()
    trend_up = recent['close'][-1] > recent['close'][0]

    levels = []
    for ratio, label in zip([0.382, 0.5, 0.618], ["38.2%", "50%", "61.8%"]):
        if trend_up:
            level = high - (high - low) * ratio
        else:
            level = low + (high - low) * ratio
        levels.append((level, label))
    return levels

def generate_binary_signals(df: pd.DataFrame, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Генерирует точки входа на основе индикаторов и уровней Фибоначчи"""
    signals = []
    
    # Уровни Фибоначчи
    fib_levels = find_fibonacci_levels(df)
    for level, label in fib_levels:
        signals.append({
            'direction': 'BUY' if df['close'].iloc[-1] > level else 'SELL',
            'price': level,
            'type': f'Уровень Фибо {label}',
            'confidence': 2,
            'expiration': '4H',
            'emoji': '📊'
        })
    
    # Сигналы от индикаторов
    if indicators['rsi'][-1] < 30:
        signals.append({
            'direction': 'BUY',
            'price': df['close'].iloc[-1],
            'type': 'RSI < 30 (Перепроданность)',
            'confidence': 3,
            'expiration': '1H',
            'emoji': '📉'
        })
    
    if indicators['rsi'][-1] > 70:
        signals.append({
            'direction': 'SELL',
            'price': df['close'].iloc[-1],
            'type': 'RSI > 70 (Перекупленность)',
            'confidence': 3,
            'expiration': '1H',
            'emoji': '📈'
        })
    
    return signals

# ================== ФУНКЦИИ ДЛЯ РАСЧЕТА ИНДИКАТОРОВ ==================
def calculate_indicators(df: pd.DataFrame, user_id: int) -> Optional[Dict[str, Any]]:
    try:
        settings = get_user_settings(user_id)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_ = df['open'].values
        
        min_period = max(settings.values())
        if len(close) < min_period:
            raise ValueError(f"Недостаточно данных для расчета. Требуется минимум {min_period} точек.")
        
        # RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(settings['RSI_PERIOD']).mean().values
        avg_loss = pd.Series(loss).rolling(settings['RSI_PERIOD']).mean().values
        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))
        
        # MACD
        ema12 = pd.Series(close).ewm(span=settings['MACD_FAST'], adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=settings['MACD_SLOW'], adjust=False).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=settings['MACD_SIGNAL'], adjust=False).mean().values
        
        # Bollinger Bands
        sma = pd.Series(close).rolling(settings['BB_PERIOD']).mean().values
        std = pd.Series(close).rolling(settings['BB_PERIOD']).std().values
        upper = sma + (std * settings['BB_STDDEV'])
        lower = sma - (std * settings['BB_STDDEV'])
        
        # Stochastic
        stoch_k = np.zeros_like(close)
        for i in range(settings['STOCH_K']-1, len(close)):
            window_high = high[i-settings['STOCH_K']+1:i+1].max()
            window_low = low[i-settings['STOCH_K']+1:i+1].min()
            stoch_k[i] = 100 * (close[i] - window_low) / (window_high - window_low + 1e-10)
        stoch_d = pd.Series(stoch_k).rolling(settings['STOCH_D']).mean().values
        
        # ADX
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), high - np.roll(high, 1), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.roll(low, 1) - low, 0)
        plus_di = 100 * pd.Series(plus_dm).rolling(settings['ADX_PERIOD']).sum().values / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(settings['ADX_PERIOD']).sum().values / tr
        adx = pd.Series(100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(settings['ADX_PERIOD']).mean().values
        
        # Supertrend
        atr = pd.Series(high - low).rolling(settings['SUPERTREND_PERIOD']).mean().values
        hl2 = (high + low) / 2
        supertrend = np.zeros_like(close)
        trend = 1
        
        for i in range(1, len(close)):
            if close[i] > hl2[i-1] + (settings['SUPERTREND_MULTIPLIER'] * atr[i-1]):
                trend = 1
            elif close[i] < hl2[i-1] - (settings['SUPERTREND_MULTIPLIER'] * atr[i-1]):
                trend = -1
            
            if trend == 1:
                supertrend[i] = hl2[i] - (settings['SUPERTREND_MULTIPLIER'] * atr[i])
            else:
                supertrend[i] = hl2[i] + (settings['SUPERTREND_MULTIPLIER'] * atr[i])
        
        # Ichimoku Cloud
        tenkan = (pd.Series(high).rolling(settings['ICHIMOKU_TENKAN']).max() + 
                pd.Series(low).rolling(settings['ICHIMOKU_TENKAN']).min()) / 2
        kijun = (pd.Series(high).rolling(settings['ICHIMOKU_KIJUN']).max() + 
               pd.Series(low).rolling(settings['ICHIMOKU_KIJUN']).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(settings['ICHIMOKU_KIJUN'])
        senkou_b = ((pd.Series(high).rolling(settings['ICHIMOKU_SENKOU']).max() + 
                   pd.Series(low).rolling(settings['ICHIMOKU_SENKOU']).min()) / 2).shift(settings['ICHIMOKU_KIJUN'])
        
        # EMA Trend Filter
        ema50 = pd.Series(close).ewm(span=settings['EMA_FAST'], adjust=False).mean().values
        ema200 = pd.Series(close).ewm(span=settings['EMA_SLOW'], adjust=False).mean().values
        
        # Свечные паттерны
        patterns = []
        if len(close) >= 3:
            o1, h1, l1, c1 = open_[-3], high[-3], low[-3], close[-3]
            o2, h2, l2, c2 = open_[-2], high[-2], low[-2], close[-2]
            o3, h3, l3, c3 = open_[-1], high[-1], low[-1], close[-1]
            
            if CANDLE_PATTERNS['hammer'](o3, h3, l3, c3):
                patterns.append('hammer')
            if CANDLE_PATTERNS['inverse_hammer'](o3, h3, l3, c3):
                patterns.append('inverse_hammer')
            if CANDLE_PATTERNS['bullish_engulfing'](o2, h2, l2, c2, o3, h3, l3, c3):
                patterns.append('bullish_engulfing')
            if CANDLE_PATTERNS['bearish_engulfing'](o2, h2, l2, c2, o3, h3, l3, c3):
                patterns.append('bearish_engulfing')
            if CANDLE_PATTERNS['morning_star'](o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
                patterns.append('morning_star')
        
        return {
            'rsi': np.concatenate(([np.nan], rsi)),
            'macd': macd,
            'signal': signal,
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx,
            'supertrend': supertrend,
            'supertrend_trend': trend,
            'tenkan': tenkan.values,
            'kijun': kijun.values,
            'senkou_a': senkou_a.values,
            'senkou_b': senkou_b.values,
            'ema50': ema50,
            'ema200': ema200,
            'patterns': patterns,
            'atr': atr,
            'close': close,
            'open': open_,
            'high': high,
            'low': low
        }
    except Exception as e:
        logger.error(f"Ошибка расчета индикаторов: {str(e)}\n{traceback.format_exc()}")
        return None

# ================== ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ СИГНАЛОВ ==================
def generate_signal(close: np.ndarray, indicators: Dict[str, Any]) -> tuple:
    if indicators is None:
        return "Ошибка расчета индикаторов", "⚠️", 0
    
    signals = []
    buy_signals = sell_signals = strength = 0
    
    # Проверка тренда по EMA
    ema50 = indicators['ema50'][-1]
    ema200 = indicators['ema200'][-1]
    trend_direction = "UP" if ema50 > ema200 else "DOWN"
    
    # RSI
    last_rsi = indicators['rsi'][-1]
    if not np.isnan(last_rsi):
        if last_rsi > 70:
            signals.append("RSI > 70 (Перекупленность)")
            sell_signals += 1
            strength += 1
        elif last_rsi < 30:
            signals.append("RSI < 30 (Перепроданность)")
            buy_signals += 1
            strength += 1
    
    # MACD
    if len(indicators['macd']) > 1 and len(indicators['signal']) > 1:
        if indicators['macd'][-1] > indicators['signal'][-1] and indicators['macd'][-2] <= indicators['signal'][-2]:
            signals.append("MACD пересек сигнал снизу вверх")
            buy_signals += 1
            strength += 1
        elif indicators['macd'][-1] < indicators['signal'][-1] and indicators['macd'][-2] >= indicators['signal'][-2]:
            signals.append("MACD пересек сигнал сверху вниз")
            sell_signals += 1
            strength += 1
    
    # Bollinger Bands
    if len(close) > 0 and 'upper' in indicators and 'lower' in indicators:
        if close[-1] < indicators['lower'][-1]:
            signals.append("Цена ниже нижней полосы Боллинджера")
            buy_signals += 1
            strength += 2
        elif close[-1] > indicators['upper'][-1]:
            signals.append("Цена выше верхней полосы Боллинджера")
            sell_signals += 1
            strength += 2
    
    # Stochastic
    if 'stoch_k' in indicators and 'stoch_d' in indicators:
        if indicators['stoch_k'][-1] < 20 and indicators['stoch_d'][-1] < 20:
            signals.append("Stochastic в зоне перепроданности")
            buy_signals += 1
            strength += 1
        elif indicators['stoch_k'][-1] > 80 and indicators['stoch_d'][-1] > 80:
            signals.append("Stochastic в зоне перекупленности")
            sell_signals += 1
            strength += 1
    
    # ADX
    if 'adx' in indicators and 'plus_di' in indicators and 'minus_di' in indicators:
        if indicators['adx'][-1] > 25:
            if indicators['plus_di'][-1] > indicators['minus_di'][-1]:
                signals.append("ADX > 25 с восходящим трендом")
                buy_signals += 1
                strength += 2
            elif indicators['plus_di'][-1] < indicators['minus_di'][-1]:
                signals.append("ADX > 25 с нисходящим трендом")
                sell_signals += 1
                strength += 2
    
    # Supertrend
    if 'supertrend_trend' in indicators:
        if indicators['supertrend_trend'] == 1:
            signals.append("Supertrend: восходящий тренд")
            buy_signals += 1
            strength += 2
        else:
            signals.append("Supertrend: нисходящий тренд")
            sell_signals += 1
            strength += 2
    
    # Ichimoku Cloud
    if 'senkou_a' in indicators and 'senkou_b' in indicators:
        if close[-1] > indicators['senkou_a'][-1] and close[-1] > indicators['senkou_b'][-1]:
            signals.append("Цена выше облака Ichimoku")
            buy_signals += 1
            strength += 2
        elif close[-1] < indicators['senkou_a'][-1] and close[-1] < indicators['senkou_b'][-1]:
            signals.append("Цена ниже облака Ichimoku")
            sell_signals += 1
            strength += 2
    
    # Свечные паттерны
    for pattern in indicators.get('patterns', []):
        if pattern in ['hammer', 'inverse_hammer', 'bullish_engulfing', 'morning_star']:
            signals.append(f"Свечной паттерн: {pattern} (бычий)")
            buy_signals += 1
            strength += 1
        elif pattern in ['bearish_engulfing']:
            signals.append(f"Свечной паттерн: {pattern} (медвежий)")
            sell_signals += 1
            strength += 1
    
    # Фильтр по тренду (EMA50/EMA200)
    if trend_direction == "UP" and sell_signals > 0:
        signals.append("⚠️ Фильтр тренда: восходящий тренд (EMA50 > EMA200)")
        sell_signals = max(0, sell_signals - 1)
        strength = max(0, strength - 1)
    elif trend_direction == "DOWN" and buy_signals > 0:
        signals.append("⚠️ Фильтр тренда: нисходящий тренд (EMA50 < EMA200)")
        buy_signals = max(0, buy_signals - 1)
        strength = max(0, strength - 1)
    
    # Фильтры
    if not is_good_time_to_trade():
        signals.append("Не лучшее время для торговли - сигнал ослаблен")
        strength = max(0, strength - 1)
    
    if 'atr' in indicators and len(indicators['atr']) > 20:
        atr_ma = np.mean(indicators['atr'][-20:])
        if indicators['atr'][-1] < atr_ma * 0.7:
            signals.append("Низкая волатильность - сигнал ослаблен")
            strength = max(0, strength - 1)
    
    # Требуем подтверждения от 3+ разных типов индикаторов
    required_confirmations = 3
    if buy_signals >= required_confirmations and strength >= 4 and trend_direction == "UP":
        return "Сильный подтвержденный сигнал на ПОКУПКУ", "🟢💪", min(strength, 5)
    elif sell_signals >= required_confirmations and strength >= 4 and trend_direction == "DOWN":
        return "Сильный подтвержденный сигнал на ПРОДАЖУ", "🔴💪", min(strength, 5)
    elif buy_signals >= required_confirmations and strength >= 3:
        return "Сильный сигнал на ПОКУПКУ", "🟢", strength
    elif sell_signals >= required_confirmations and strength >= 3:
        return "Сильный сигнал на ПРОДАЖУ", "🔴", strength
    elif buy_signals >= 2 and strength >= 2:
        return "Средний сигнал на ПОКУПКУ", "🟡", strength
    elif sell_signals >= 2 and strength >= 2:
        return "Средний сигнал на ПРОДАЖУ", "🟠", strength
    elif buy_signals >= 1:
        return "Слабый сигнал на ПОКУПКУ", "🔵", strength
    elif sell_signals >= 1:
        return "Слабый сигнал на ПРОДАЖУ", "🟣", strength
    
    return "Смешанные сигналы", "⚪", strength

def is_good_time_to_trade() -> bool:
    hour = datetime.now().hour
    return (8 <= hour <= 12) or (14 <= hour <= 20)

def get_multi_timeframe_analysis(symbol: str, user_id: int) -> Dict[str, Any]:
    analysis = {}
    for timeframe in MULTI_TIMEFRAMES:
        df = get_ohlc_data(symbol, timeframe)
        analysis[timeframe] = calculate_indicators(df, user_id)
    return analysis

def check_multi_timeframe_confirmation(symbol: str, user_id: int, direction: str) -> bool:
    analysis = get_multi_timeframe_analysis(symbol, user_id)
    confirmations = 0
    
    for timeframe in MULTI_TIMEFRAMES:
        indicators = analysis[timeframe]
        signal, _, _ = generate_signal(indicators['close'], indicators)
        
        if direction == "BUY" and "BUY" in signal:
            confirmations += 1
        elif direction == "SELL" and "SELL" in signal:
            confirmations += 1
    
    return confirmations >= 2

def optimize_parameters(symbol: str, timeframe: str, user_id: int) -> Dict[str, Any]:
    user_data = ensure_user_data(user_id)
    cache_key = f"{symbol}_{timeframe}"
    if cache_key in user_data['optimized_params']:
        return user_data['optimized_params'][cache_key]
    
    df = get_ohlc_data(symbol, timeframe)
    if len(df) < 100:
        return DEFAULT_SETTINGS
    
    best_settings = DEFAULT_SETTINGS.copy()
    best_score = 0
    
    # Тестируем разные комбинации параметров
    for rsi_period in [12, 14, 16]:
        for bb_period in [18, 20, 22]:
            for stoch_k in [12, 14, 16]:
                current_settings = DEFAULT_SETTINGS.copy()
                current_settings['RSI_PERIOD'] = rsi_period
                current_settings['BB_PERIOD'] = bb_period
                current_settings['STOCH_K'] = stoch_k
                
                # Тестируем на исторических данных
                indicators = calculate_indicators(df, user_id)
                signals = []
                for i in range(100, len(df)):
                    signal, _, strength = generate_signal(df['close'].values[:i], indicators)
                    if "BUY" in signal or "SELL" in signal:
                        signals.append((signal, strength, df['close'].values[i]))
                
                # Оцениваем качество сигналов
                if len(signals) > 5:
                    score = sum(s[1] for s in signals) / len(signals)
                    if score > best_score:
                        best_score = score
                        best_settings = current_settings
    
    user_data['optimized_params'][cache_key] = best_settings
    return best_settings

# ================== ФУНКЦИИ ДЛЯ СОЗДАНИЯ ГРАФИКОВ ==================
def create_candlestick_plot(df: pd.DataFrame, indicators: Dict[str, Any], symbol: str, timeframe: str) -> io.BytesIO:
    try:
        # Стилизованные цвета
        mc = mpf.make_marketcolors(
            up=COLORS['up'],
            down=COLORS['down'],
            wick={'up':COLORS['wick_up'], 'down':COLORS['wick_down']},
            edge={'up':COLORS['wick_up'], 'down':COLORS['wick_down']}
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            gridcolor=COLORS['grid'],
            facecolor=COLORS['background'],
            edgecolor='#dddddd',
            figcolor='white'
        )
        
        # Добавляем индикаторы
        add_plot = [
            mpf.make_addplot(indicators['ema50'], color=COLORS['ema_fast'], width=1.5),
            mpf.make_addplot(indicators['ema200'], color=COLORS['ema_slow'], width=1.5),
            mpf.make_addplot(indicators['supertrend'], color=COLORS['supertrend'], width=1)
        ]
        
        buf = io.BytesIO()
        fig, _ = mpf.plot(
            df,
            type='candle',
            style=s,
            addplot=add_plot,
            figscale=1.1,
            figratio=(10, 6),
            title=f"\n\n{symbol} ({timeframe})",
            returnfig=True
        )
        
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logger.error(f"Ошибка создания свечного графика: {str(e)}\n{traceback.format_exc()}")
        return None

def create_enhanced_plot(df: pd.DataFrame, indicators: Dict[str, Any]) -> io.BytesIO:
    try:
        # Ручная настройка стиля вместо seaborn
        plt.rcParams.update({
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            'axes.edgecolor': '#dddddd',
            'axes.linewidth': 0.8,
            'figure.facecolor': 'white'
        })
        
        plt.figure(figsize=(14, 22), dpi=120)
        
        # 1. Price with Ichimoku Cloud and EMAs
        ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=2)
        ax1.plot(df.index, df['close'], label='Цена', color=COLORS['text'], linewidth=1.5)
        ax1.plot(df.index, indicators['ema50'], label='EMA50', color=COLORS['ema_fast'], linestyle='--', alpha=0.8)
        ax1.plot(df.index, indicators['ema200'], label='EMA200', color=COLORS['ema_slow'], linestyle='--', alpha=0.8)
        
        # Ichimoku Cloud
        ax1.fill_between(df.index, indicators['senkou_a'], indicators['senkou_b'], 
                        where=indicators['senkou_a'] >= indicators['senkou_b'], 
                        facecolor='#2ecc71', alpha=0.2, label='Облако Ichimoku (Bullish)')
        ax1.fill_between(df.index, indicators['senkou_a'], indicators['senkou_b'], 
                        where=indicators['senkou_a'] < indicators['senkou_b'], 
                        facecolor='#e74c3c', alpha=0.2, label='Облако Ichimoku (Bearish)')
        ax1.plot(df.index, indicators['tenkan'], label='Tenkan', color='#3498db', linestyle=':')
        ax1.plot(df.index, indicators['kijun'], label='Kijun', color='#9b59b6', linestyle=':')
        
        ax1.set_title('Цена с индикаторами', fontsize=12, pad=20, color=COLORS['text'])
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. MACD
        ax3 = plt.subplot2grid((7, 1), (2, 0))
        ax3.plot(df.index, indicators['macd'], label='MACD', color=COLORS['ema_fast'])
        ax3.plot(df.index, indicators['signal'], label='Signal', color=COLORS['ema_slow'])
        ax3.fill_between(df.index, indicators['macd'], indicators['signal'],
                        where=indicators['macd']>indicators['signal'],
                        facecolor=COLORS['up'], alpha=0.3)
        ax3.fill_between(df.index, indicators['macd'], indicators['signal'],
                        where=indicators['macd']<=indicators['signal'],
                        facecolor=COLORS['down'], alpha=0.3)
        ax3.axhline(0, color=COLORS['text'], linestyle='--', linewidth=0.5)
        ax3.set_title('MACD', fontsize=12, pad=10, color=COLORS['text'])
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 3. RSI
        ax4 = plt.subplot2grid((7, 1), (3, 0))
        ax4.plot(df.index, indicators['rsi'], label='RSI', color='#9b59b6')
        ax4.axhline(70, linestyle='--', color=COLORS['down'], alpha=0.5)
        ax4.axhline(30, linestyle='--', color=COLORS['up'], alpha=0.5)
        ax4.fill_between(df.index, indicators['rsi'], 70, where=indicators['rsi']>=70,
                        facecolor=COLORS['down'], alpha=0.1)
        ax4.fill_between(df.index, indicators['rsi'], 30, where=indicators['rsi']<=30,
                        facecolor=COLORS['up'], alpha=0.1)
        ax4.set_title('RSI', fontsize=12, pad=10, color=COLORS['text'])
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 4. Stochastic
        ax5 = plt.subplot2grid((7, 1), (4, 0))
        ax5.plot(df.index, indicators['stoch_k'], label='%K', color='#3498db')
        ax5.plot(df.index, indicators['stoch_d'], label='%D', color='#e74c3c')
        ax5.axhline(80, linestyle='--', color=COLORS['down'], alpha=0.5)
        ax5.axhline(20, linestyle='--', color=COLORS['up'], alpha=0.5)
        ax5.fill_between(df.index, indicators['stoch_k'], 80, where=indicators['stoch_k']>=80,
                        facecolor=COLORS['down'], alpha=0.1)
        ax5.fill_between(df.index, indicators['stoch_k'], 20, where=indicators['stoch_k']<=20,
                        facecolor=COLORS['up'], alpha=0.1)
        ax5.set_title('Stochastic', fontsize=12, pad=10, color=COLORS['text'])
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 5. ADX
        ax6 = plt.subplot2grid((7, 1), (5, 0))
        ax6.plot(df.index, indicators['adx'], label='ADX', color=COLORS['text'])
        ax6.plot(df.index, indicators['plus_di'], label='+DI', color=COLORS['up'])
        ax6.plot(df.index, indicators['minus_di'], label='-DI', color=COLORS['down'])
        ax6.axhline(25, linestyle='--', color='#3498db', alpha=0.5)
        ax6.set_title('ADX', fontsize=12, pad=10, color=COLORS['text'])
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 6. Supertrend
        ax7 = plt.subplot2grid((7, 1), (6, 0))
        ax7.plot(df.index, df['close'], label='Цена', color=COLORS['text'], linewidth=1)
        ax7.plot(df.index, indicators['supertrend'], 
                label='Supertrend', 
                color=COLORS['up'] if indicators['supertrend_trend'] == 1 else COLORS['down'])
        ax7.set_title('Supertrend', fontsize=12, pad=10, color=COLORS['text'])
        ax7.legend(loc='upper left', fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logger.error(f"Ошибка создания графика индикаторов: {str(e)}\n{traceback.format_exc()}")
        return None

# ================== ОСНОВНЫЕ ФУНКЦИИ БОТА ==================
def start(update: Update, context: CallbackContext) -> None:
    if not update.effective_user or update.effective_user.is_bot:
        return
    
    user_id = update.effective_user.id
    if not check_access(user_id):
        context.bot.send_message(chat_id=user_id, text="⛔ У вас нет доступа к этому боту")
        return
    
    cleanup_messages(context, update.effective_chat.id, user_id)
    ensure_user_data(user_id)['step'] = 'main_menu'
    
    buttons = [
        ("📊 Начать анализ", "start_analysis"),
        ("⚙️ Настройки", "settings_menu"),
        ("ℹ️ Помощь", "help"),
        ("🔄 Сбросить настройки", "reset_settings")
    ]
    
    msg_id = safe_send_message(
        context,
        update.effective_chat.id,
        "📈 <b>Forex Trading Bot</b>\nВыберите действие:",
        reply_markup=create_menu_buttons(buttons, back_button=False)
    )
    
    if msg_id:
        add_message_to_cleanup(user_id, msg_id)

def help_command(update: Update, context: CallbackContext) -> None:
    if not update.effective_user or not check_access(update.effective_user.id):
        return
    
    text = (
        "ℹ️ <b>Помощь по боту</b>\n\n"
        "Этот бот предоставляет технический анализ финансовых инструментов.\n\n"
        "<b>Основные функции:</b>\n"
        "- Анализ валютных пар и криптовалют\n"
        "- Настройка параметров индикаторов\n"
        "- Различные таймфреймы (5M, 15M, 1H, 4H)\n"
        "- Визуализация данных с графиками\n\n"
        "<b>Доступные команды:</b>\n"
        "/start - Главное меню\n"
        "/help - Эта справка"
    )
    
    msg_id = safe_send_message(
        context,
        update.effective_chat.id,
        text,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 В главное меню", callback_data='back_to_main')]
        ])
    )
    
    if msg_id:
        add_message_to_cleanup(update.effective_user.id, msg_id)

def button_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    if not query or not query.from_user or query.from_user.is_bot or not check_access(query.from_user.id):
        return
    
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    message_id = query.message.message_id
    
    if not safe_answer_query(query):
        return
    
    user_data = ensure_user_data(user_id)
    current_step = user_data.get('step', '')
    
    if query.data == 'start_analysis':
        buttons = [(f"🇪🇺 EUR", 'currency_EUR'), 
                  (f"🇺🇸 USD", 'currency_USD'), 
                  (f"🇬🇧 GBP", 'currency_GBP'),
                  (f"🇦🇺 AUD", 'currency_AUD'),
                  (f"🇯🇵 JPY", 'currency_JPY'),
                  (f"🇨🇦 CAD", 'currency_CAD'),
                  (f"🇨🇭 CHF", 'currency_CHF'),
                  (f"₿ BTC", 'currency_BTC')]
        
        if not safe_edit_message(context, chat_id, message_id, "📊 <b>Выберите базовую валюту:</b>", create_menu_buttons(buttons, back_button=False)):
            cleanup_and_send(context, chat_id, user_id, "📊 <b>Выберите базовую валюту:</b>", buttons)
        user_data['step'] = 'select_currency'
    
    elif query.data.startswith('currency_'):
        currency = query.data.split('_')[1]
        pairs = CURRENCY_GROUPS.get(currency, [])
        buttons = [(pair, f'pair_{i}') for i, pair in enumerate(pairs)]
        if not safe_edit_message(context, chat_id, message_id, f"📊 <b>Выберите валютную пару ({currency}):</b>", create_menu_buttons(buttons)):
            cleanup_and_send(context, chat_id, user_id, f"📊 <b>Выберите валютную пару ({currency}):</b>", buttons)
        user_data['step'] = 'select_pair'
        user_data['currency'] = currency
    
    elif query.data.startswith('pair_'):
        pair_idx = int(query.data.split('_')[1])
        currency = user_data.get('currency', '')
        if pair_idx < len(CURRENCY_GROUPS.get(currency, [])):
            user_data['symbol'] = CURRENCY_GROUPS[currency][pair_idx]
            buttons = [(tf, f'tf_{i}') for i, tf in enumerate(TIMEFRAMES)]
            if not safe_edit_message(context, chat_id, message_id, f"📊 <b>Выбрана пара: {user_data['symbol']}</b>\nВыберите таймфрейм:", create_menu_buttons(buttons)):
                cleanup_and_send(context, chat_id, user_id, f"📊 <b>Выбрана пара: {user_data['symbol']}</b>\nВыберите таймфрейм:", buttons)
            user_data['step'] = 'select_timeframe'
        else:
            safe_answer_query(query, "Ошибка выбора валютной пары", True)
    
    elif query.data.startswith('tf_'):
        tf_idx = int(query.data.split('_')[1])
        if tf_idx < len(TIMEFRAMES):
            process_timeframe_selection(update, context, user_data, user_id, chat_id, tf_idx)
        else:
            safe_answer_query(query, "Ошибка выбора таймфрейма", True)
    
    elif query.data == 'back':
        handle_back_action(update, context)
    elif query.data == 'back_to_main':
        start(update, context)
    elif query.data == 'settings_menu':
        show_settings_menu(update, context, user_id)
    elif query.data == 'reset_settings':
        user_data['settings'] = DEFAULT_SETTINGS.copy()
        user_data['optimized_params'] = {}
        safe_answer_query(query, "Настройки сброшены к значениям по умолчанию")
        show_settings_menu(update, context, user_id)
    elif query.data.startswith('set_'):
        handle_setting_selection(query, context, user_data, chat_id, message_id)

def process_timeframe_selection(update, context, user_data, user_id, chat_id, tf_idx):
    context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
    cleanup_messages(context, chat_id, user_id)

    try:
        symbol = user_data.get('symbol', '')
        timeframe = TIMEFRAMES[tf_idx]
        user_data['timeframe'] = timeframe

        df = get_ohlc_data(symbol, timeframe)
        indicators = calculate_indicators(df, user_id)
        if indicators is None:
            raise ValueError("Не удалось рассчитать индикаторы")

        # Основной сигнал
        signal_text, emoji, strength = generate_signal(df['close'].values, indicators)
        if "BUY" in signal_text and not check_multi_timeframe_confirmation(symbol, user_id, "BUY"):
            signal_text = "Неподтвержденный " + signal_text
            strength = max(0, strength - 1)
        elif "SELL" in signal_text and not check_multi_timeframe_confirmation(symbol, user_id, "SELL"):
            signal_text = "Неподтвержденный " + signal_text
            strength = max(0, strength - 1)

        # Дополнительные точки входа
        binary_signals = generate_binary_signals(df, indicators)
        entry_block = "\n📌 <b>Точки входа (включая Фибоначчи):</b>\n"
        for s in binary_signals:
            if s['direction']:
                entry_block += (
                    f"{s['emoji']} <b>{s['direction']}</b> @ {s['price']:.5f}\n"
                    f"⏱ {s['expiration']} | ★{'★'*s['confidence']}\n"
                    f"📊 {s['type']}\n━━━━━━\n"
                )
            else:
                entry_block += f"{s['emoji']} {s['type']}\n"

        text = (
            f"📊 <b>{symbol} | {timeframe}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🔼 <b>Цена:</b> {df['close'].iloc[-1]:.5f}\n"
            f"🔽 <b>Изменение (24ч):</b> "
            f"{((df['close'].iloc[-1] - df['close'].iloc[-24])/df['close'].iloc[-24]*100):+.2f}%\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🚀 <b>Сигнал:</b> {emoji} {signal_text}\n"
            f"{entry_block}"
        )

        candle_plot = create_candlestick_plot(df, indicators, symbol, timeframe)
        tech_plot = create_enhanced_plot(df, indicators)

        msg1 = context.bot.send_photo(chat_id, candle_plot, caption=f"<b>{symbol} ({timeframe}) - Свечной график</b>", parse_mode='HTML')
        add_message_to_cleanup(user_id, msg1.message_id)

        msg2 = context.bot.send_photo(
            chat_id,
            tech_plot,
            caption=text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Обновить", callback_data=f'tf_{tf_idx}')],
                [InlineKeyboardButton("🔙 Назад", callback_data=f'currency_{user_data.get("currency", "")}')]
            ])
        )
        add_message_to_cleanup(user_id, msg2.message_id)

    except Exception as e:
        logger.error(f"Ошибка анализа: {str(e)}\n{traceback.format_exc()}")
        msg_id = safe_send_message(
            context,
            chat_id,
            "⚠️ Ошибка при анализе. Попробуйте позже.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Повторить попытку", callback_data=f'tf_{tf_idx}')],
                [InlineKeyboardButton("🔙 Назад", callback_data='back')]
            ])
        )
        if msg_id:
            add_message_to_cleanup(user_id, msg_id)

def show_settings_menu(update: Update, context: CallbackContext, user_id: int) -> None:
    settings = ensure_user_data(user_id)['settings']
    
    buttons = [
        [
            InlineKeyboardButton(f"📊 RSI: {settings['RSI_PERIOD']}", callback_data='set_rsi'),
            InlineKeyboardButton(f"📉 MACD: {settings['MACD_FAST']}/{settings['MACD_SLOW']}", callback_data='set_macd')
        ],
        [
            InlineKeyboardButton(f"📊 BBands: {settings['BB_PERIOD']}/{settings['BB_STDDEV']}", callback_data='set_bb'),
            InlineKeyboardButton(f"📈 Stochastic: {settings['STOCH_K']}/{settings['STOCH_D']}", callback_data='set_stoch')
        ],
        [
            InlineKeyboardButton(f"📊 ADX: {settings['ADX_PERIOD']}", callback_data='set_adx'),
            InlineKeyboardButton(f"📉 Supertrend: {settings['SUPERTREND_PERIOD']}/{settings['SUPERTREND_MULTIPLIER']}", callback_data='set_supertrend')
        ],
        [
            InlineKeyboardButton(f"📊 Ichimoku: {settings['ICHIMOKU_TENKAN']}/{settings['ICHIMOKU_KIJUN']}", callback_data='set_ichimoku'),
            InlineKeyboardButton(f"📈 EMA: {settings['EMA_FAST']}/{settings['EMA_SLOW']}", callback_data='set_ema')
        ],
        [InlineKeyboardButton("🔙 В главное меню", callback_data='back_to_main')]
    ]
    
    safe_edit_message(
        context,
        update.callback_query.message.chat_id,
        update.callback_query.message.message_id,
        "⚙️ <b>Настройки индикаторов</b>\nВыберите параметр для настройки:",
        InlineKeyboardMarkup(buttons)
    )
    ensure_user_data(user_id)['step'] = 'settings_menu'

def handle_setting_selection(query, context, user_data, chat_id, message_id):
    setting = query.data.split('_')[1]
    user_data['step'] = f'set_{setting}'
    
    prompts = {
        'rsi': "Введите период RSI (5-30):",
        'macd': "Введите периоды MACD в формате быстрый/медленный/сигнал (например, 12/26/9):",
        'bb': "Введите период и стандартное отклонение Bollinger Bands в формате период/отклонение (например, 20/2):",
        'stoch': "Введите периоды Stochastic в формате %K/%D (например, 14/3):",
        'adx': "Введите период ADX (5-30):",
        'supertrend': "Введите период и множитель Supertrend в формате период/множитель (например, 10/3.0):",
        'ichimoku': "Введите периоды Ichimoku в формате Tenkan/Kijun/Senkou (например, 9/26/52):",
        'ema': "Введите периоды EMA в формате быстрая/медленная (например, 50/200):"
    }
    
    if not safe_edit_message(
        context,
        chat_id,
        message_id,
        prompts.get(setting, "Введите новое значение параметра:"),
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data='settings_menu')]
        ])
    ):
        cleanup_and_send(
            context,
            chat_id,
            query.from_user.id,
            prompts.get(setting, "Введите новое значение параметра:"),
            [("🔙 Назад", 'settings_menu')]
        )

def handle_back_action(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = query.from_user.id
    user_data = ensure_user_data(user_id)
    current_step = user_data.get('step', '')
    
    if current_step == 'select_pair':
        buttons = [(f"🇪🇺 EUR", 'currency_EUR'), 
                  (f"🇺🇸 USD", 'currency_USD'), 
                  (f"🇬🇧 GBP", 'currency_GBP'),
                  (f"🇦🇺 AUD", 'currency_AUD'),
                  (f"🇯🇵 JPY", 'currency_JPY'),
                  (f"🇨🇦 CAD", 'currency_CAD'),
                  (f"🇨🇭 CHF", 'currency_CHF'),
                  (f"₿ BTC", 'currency_BTC')]
        
        if not safe_edit_message(
            context,
            query.message.chat_id,
            query.message.message_id,
            "📊 <b>Выберите базовую валюту:</b>",
            create_menu_buttons(buttons, back_button=False)
        ):
            cleanup_and_send(
                context,
                query.message.chat_id,
                user_id,
                "📊 <b>Выберите базовую валюту:</b>",
                buttons
            )
        user_data['step'] = 'select_currency'
    
    elif current_step == 'select_timeframe':
        currency = user_data.get('currency', '')
        pairs = CURRENCY_GROUPS.get(currency, [])
        buttons = [(pair, f'pair_{i}') for i, pair in enumerate(pairs)]
        
        if not safe_edit_message(
            context,
            query.message.chat_id,
            query.message.message_id,
            f"📊 <b>Выберите валютную пару ({currency}):</b>",
            create_menu_buttons(buttons)
        ):
            cleanup_and_send(
                context,
                query.message.chat_id,
                user_id,
                f"📊 <b>Выберите валютную пару ({currency}):</b>",
                buttons
            )
        user_data['step'] = 'select_pair'
    
    elif current_step.startswith('set_'):
        show_settings_menu(update, context, user_id)
    
    elif current_step == 'settings_menu':
        start(update, context)
    
    else:
        start(update, context)

def text_handler(update: Update, context: CallbackContext) -> None:
    if not update.message or not update.message.from_user or update.message.from_user.is_bot or not check_access(update.message.from_user.id):
        return
    
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    text = update.message.text.strip()
    user_data = ensure_user_data(user_id)
    
    if user_data.get('step', '').startswith('set_'):
        handle_settings_input(update, context, user_id, text)
    else:
        cleanup_messages(context, chat_id, user_id)
        msg_id = safe_send_message(
            context,
            chat_id,
            "Используйте кнопки меню для навигации",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 В главное меню", callback_data='back_to_main')]
            ])
        )
        if msg_id:
            add_message_to_cleanup(user_id, msg_id)

def handle_settings_input(update: Update, context: CallbackContext, user_id: int, text: str) -> None:
    if not check_access(user_id):
        return
    
    chat_id = update.message.chat_id
    cleanup_messages(context, chat_id, user_id)
    user_data = ensure_user_data(user_id)
    settings = user_data['settings']
    current_step = user_data['step']
    
    setting_map = {
        'set_rsi': ('RSI_PERIOD', lambda x: x.isdigit() and 5 <= int(x) <= 30, int),
        'set_macd': (['MACD_FAST', 'MACD_SLOW', 'MACD_SIGNAL'], 
                    lambda x: len(x.split('/')) == 3 and all(i.isdigit() and 5 <= int(i) <= 50 for i in x.split('/')[:2]) and x.split('/')[2].isdigit() and 5 <= int(x.split('/')[2]) <= 20,
                    lambda x: list(map(int, x.split('/')))),
        'set_bb': (['BB_PERIOD', 'BB_STDDEV'], 
                  lambda x: len(x.split('/')) == 2 and x.split('/')[0].isdigit() and 5 <= int(x.split('/')[0]) <= 50 and x.split('/')[1].isdigit() and 1 <= int(x.split('/')[1]) <= 3,
                  lambda x: list(map(int, x.split('/')))),
        'set_stoch': (['STOCH_K', 'STOCH_D'], 
                     lambda x: len(x.split('/')) == 2 and x.split('/')[0].isdigit() and 5 <= int(x.split('/')[0]) <= 30 and x.split('/')[1].isdigit() and 1 <= int(x.split('/')[1]) <= 10,
                     lambda x: list(map(int, x.split('/')))),
        'set_adx': ('ADX_PERIOD', lambda x: x.isdigit() and 5 <= int(x) <= 30, int),
        'set_supertrend': (['SUPERTREND_PERIOD', 'SUPERTREND_MULTIPLIER'], 
                          lambda x: len(x.split('/')) == 2 and x.split('/')[0].isdigit() and 5 <= int(x.split('/')[0]) <= 30 and x.split('/')[1].replace('.', '').isdigit() and 1.0 <= float(x.split('/')[1]) <= 5.0,
                          lambda x: [int(x.split('/')[0]), float(x.split('/')[1])]),
        'set_ichimoku': (['ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU'], 
                        lambda x: len(x.split('/')) == 3 and all(i.isdigit() and 5 <= int(i) <= 100 for i in x.split('/')),
                        lambda x: list(map(int, x.split('/')))),
        'set_ema': (['EMA_FAST', 'EMA_SLOW'],
                   lambda x: len(x.split('/')) == 2 and x.split('/')[0].isdigit() and 5 <= int(x.split('/')[0]) <= 100 and x.split('/')[1].isdigit() and 50 <= int(x.split('/')[1]) <= 300,
                   lambda x: list(map(int, x.split('/'))))
    }
    
    setting_name, validation_func, conversion_func = setting_map.get(current_step, (None, None, None))
    
    try:
        if validation_func and validation_func(text):
            values = conversion_func(text) if conversion_func else text
            if isinstance(setting_name, list):
                for name, value in zip(setting_name, values):
                    settings[name] = value
            else:
                settings[setting_name] = values
                
            # Сбрасываем оптимизированные параметры при изменении настроек
            user_data['optimized_params'] = {}
                
            msg_id = safe_send_message(
                context,
                chat_id,
                f"✅ Параметр успешно обновлен",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад к настройкам", callback_data='settings_menu')]
                ])
            )
            if msg_id:
                add_message_to_cleanup(user_id, msg_id)
            user_data['step'] = 'settings_menu'
        else:
            msg_id = safe_send_message(context, chat_id, "Неверный формат. Пожалуйста, введите значение в правильном формате.")
            if msg_id:
                add_message_to_cleanup(user_id, msg_id)
    except ValueError:
        msg_id = safe_send_message(context, chat_id, "Ошибка преобразования данных. Пожалуйста, введите корректные значения.")
        if msg_id:
            add_message_to_cleanup(user_id, msg_id)

def error_handler(update: Optional[Update], context: CallbackContext, error: Exception) -> None:
    logger.error(f"Ошибка: {str(error)}\n{traceback.format_exc()}")
    
    try:
        if update and update.message:
            chat_id = update.message.chat_id
            user_id = update.message.from_user.id
            cleanup_messages(context, chat_id, user_id)
            msg_id = safe_send_message(
                context,
                chat_id,
                "⚠️ Произошла ошибка. Пожалуйста, попробуйте позже.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 В главное меню", callback_data='back_to_main')]
                ])
            )
            if msg_id:
                add_message_to_cleanup(user_id, msg_id)
        elif update and update.callback_query and update.callback_query.message:
            chat_id = update.callback_query.message.chat_id
            user_id = update.callback_query.from_user.id
            cleanup_messages(context, chat_id, user_id)
            msg_id = safe_send_message(
                context,
                chat_id,
                "⚠️ Произошла ошибка. Пожалуйста, попробуйте позже.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 В главное меню", callback_data='back_to_main')]
                ])
            )
            if msg_id:
                add_message_to_cleanup(user_id, msg_id)
    except Exception as e:
        logger.error(f"Ошибка в обработчике ошибок: {str(e)}")

def cleanup_old_users():
    now = time.time()
    for user_id in list(user_data.keys()):
        if now - user_data[user_id].get('last_activity', 0) > 86400:
            del user_data[user_id]
    for user_id in list(user_messages.keys()):
        if user_id not in user_data:
            del user_messages[user_id]

def main() -> None:
    try:
        updater = Updater('7844378262:AAHmBSGea0znCCks3hsxpixCjEYeczwalsA', use_context=True)
        
        global BOT_ID
        BOT_ID = updater.bot.id
        
        dp = updater.dispatcher

        dp.add_handler(CommandHandler("start", start))
        dp.add_handler(CommandHandler("help", help_command))
        dp.add_handler(CallbackQueryHandler(button_handler))
        dp.add_handler(MessageHandler(Filters.text & ~Filters.command, text_handler))
        dp.add_error_handler(error_handler)

        job_queue = updater.job_queue
        job_queue.run_repeating(lambda _: cleanup_old_users(), interval=3600)

        logger.info("Бот запущен")
        updater.start_polling()
        updater.idle()

    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {str(e)}\n{traceback.format_exc()}")

if __name__ == '__main__':
    main()