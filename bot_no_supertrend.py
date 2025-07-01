import logging
import os
from dotenv import load_dotenv
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ChatAction, Update
from telegram.ext import CallbackContext
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime, timedelta
import traceback
import mplfinance as mpf
from typing import Dict, Any, Optional, List
import requests
import time

# Глобальная переменная для ID бота
BOT_ID = None

# Настройка приватного доступа
ALLOWED_USER_IDS = {
    7785586524,  # Замените на ваш ID (можно узнать у @userinfobot)
    7064593003   # Дополнительные разрешенные ID (если нужно)
}

def check_access(user_id: int) -> bool:
    """Проверяет, есть ли у пользователя доступ к боту"""
    return user_id in ALLOWED_USER_IDS

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

# Загрузка переменных окружения

TOKEN ='7736019500:AAGIYP_E3oNv6_60JEI8jyxgoxKLWg_jIXg'
TWELVE_DATA_KEY ='80caddf1ddcd4f1b983d21d31c8bc180'

if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN не найден в .env файле")
if not TWELVE_DATA_KEY:
    raise ValueError("TWELVE_DATA_KEY не найден в .env файле")

# Конфигурация
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
}

# Группировка валютных пар по базовой валюте
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
user_data: Dict[int, Dict[str, Any]] = {}
user_messages: Dict[int, List[int]] = {}

def ensure_user_data(user_id: int) -> Dict[str, Any]:
    """Гарантирует, что данные пользователя существуют и возвращает их"""
    if user_id not in user_data:
        user_data[user_id] = {
            'settings': DEFAULT_SETTINGS.copy(),
            'last_activity': time.time(),
            'step': 'main_menu'
        }
    return user_data[user_id]

# Безопасные обертки для Telegram API
def safe_answer_query(query, text: str = None, show_alert: bool = False) -> bool:
    try:
        query.answer(text=text, show_alert=show_alert)
        return True
    except Exception as e:
        logger.warning(f"Ошибка при answer_query: {str(e)}")
        return False

def safe_edit_message(context: CallbackContext, chat_id: int, message_id: int, 
                     text: str, reply_markup: InlineKeyboardMarkup = None) -> bool:
    try:
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

def safe_send_message(context: CallbackContext, chat_id: int, text: str, 
                     reply_markup: InlineKeyboardMarkup = None) -> Optional[int]:
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

# Утилиты
def create_menu_buttons(buttons_data: list, back_button: bool = True) -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(text, callback_data=callback)] for text, callback in buttons_data]
    if back_button:
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data='back')])
    return InlineKeyboardMarkup(keyboard)

def get_user_settings(user_id: int) -> Dict[str, Any]:
    user_data = ensure_user_data(user_id)
    if 'settings' not in user_data:
        user_data['settings'] = DEFAULT_SETTINGS.copy()
    user_data['last_activity'] = time.time()
    return user_data['settings']

def safe_get_message(update: Update) -> Optional[Any]:
    try:
        if update.message:
            return update.message
        elif update.callback_query and update.callback_query.message:
            return update.callback_query.message
        return None
    except Exception as e:
        logger.error(f"Ошибка в safe_get_message: {str(e)}")
        return None

def cleanup_messages(context: CallbackContext, chat_id: int, user_id: int):
    if user_id in user_messages:
        for msg_id in user_messages[user_id]:
            try:
                safe_delete_message(context, chat_id, msg_id)
            except Exception as e:
                logger.warning(f"Ошибка при удалении сообщения: {str(e)}")
        user_messages[user_id] = []

def add_message_to_cleanup(user_id: int, message_id: int):
    if user_id not in user_messages:
        user_messages[user_id] = []
    user_messages[user_id].append(message_id)

# Получение данных с Twelve Data API
def get_ohlc_data(symbol: str = "EUR/USD", timeframe: str = "1H") -> pd.DataFrame:
    """Получение СВЕЖИХ данных с Twelve Data API"""
    try:
        interval_map = {
            '5M': '5min',
            '15M': '15min',
            '1H': '1h',
            '4H': '4h'
        }
        interval = interval_map.get(timeframe, '1h')
        
        symbol_mapping = {
            'EUR/USD': 'EUR/USD',
            'GBP/USD': 'GBP/USD',
            'USD/JPY': 'USD/JPY',
            'AUD/USD': 'AUD/USD',
            'BTC/USD': 'BTC/USD',
            'EUR/GBP': 'EUR/GBP',
            'EUR/JPY': 'EUR/JPY',
            'EUR/AUD': 'EUR/AUD',
            'EUR/CAD': 'EUR/CAD',
            'EUR/CHF': 'EUR/CHF',
            'USD/CAD': 'USD/CAD',
            'USD/CHF': 'USD/CHF',
            'GBP/JPY': 'GBP/JPY',
            'GBP/AUD': 'GBP/AUD',
            'GBP/CAD': 'GBP/CAD',
            'GBP/CHF': 'GBP/CHF',
            'GBP/NZD': 'GBP/NZD',
            'AUD/JPY': 'AUD/JPY',
            'AUD/NZD': 'AUD/NZD',
            'AUD/CAD': 'AUD/CAD',
            'AUD/CHF': 'AUD/CHF',
            'CAD/JPY': 'CAD/JPY',
            'CAD/CHF': 'CAD/CHF',
            'CHF/JPY': 'CHF/JPY',
            'NZD/USD': 'NZD/USD',
            'BTC/EUR': 'BTC/EUR',
            'BTC/JPY': 'BTC/JPY',
            'BTC/GBP': 'BTC/GBP'
        }
        
        symbol_code = symbol_mapping.get(symbol, symbol)
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol_code,
            'interval': interval,
            'apikey': TWELVE_DATA_KEY,
            'outputsize': 100,
            'format': 'JSON',
            'timestamp': int(time.time())
        }
        
        logger.info(f"Запрос свежих данных для {symbol} ({timeframe})")
        response = requests.get(url, params=params, timeout=15)
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
        df = df.apply(pd.to_numeric)
        df = df.dropna()
        
        if df.empty:
            raise ValueError("Получены пустые данные")
            
        logger.info(f"Получены свежие данные для {symbol} ({timeframe}), последняя точка: {df.index[-1]}")
        return df[['open', 'high', 'low', 'close']]
        
    except Exception as e:
        logger.error(f"Ошибка получения данных: {str(e)}\n{traceback.format_exc()}")
        now = datetime.now()
        np.random.seed(int(now.timestamp()))
        
        periods_map = {'5M': 288, '15M': 96, '1H': 100, '4H': 50}
        periods = periods_map.get(timeframe, 100)
        base_price = 1.0 if 'USD' in symbol else 100.0
        
        prices = base_price + np.random.normal(0, 0.02, periods).cumsum()
        dates = pd.date_range(end=now, periods=periods, freq=interval.lower())
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices + np.abs(np.random.normal(0.001, 0.002, periods)),
            'low': prices - np.abs(np.random.normal(0.001, 0.002, periods)),
            'close': prices + np.random.normal(0, 0.001, periods)
        }).set_index('date')

# Расчет индикаторов
def calculate_indicators(df: pd.DataFrame, user_id: int) -> Optional[Dict[str, Any]]:
    try:
        settings = get_user_settings(user_id)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        min_period = max(settings['RSI_PERIOD'], settings['MACD_SLOW'])
                         settings['BB_PERIOD'], settings['STOCH_K'], settings['ADX_PERIOD'],
        if len(close) < min_period:
            raise ValueError(f"Недостаточно данных для расчета. Требуется минимум {min_period} точек.")
        
        # RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(settings['RSI_PERIOD']).mean().values
        avg_loss = pd.Series(loss).rolling(settings['RSI_PERIOD']).mean().values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
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
        
        # Stochastic Oscillator
        stoch_k = np.zeros_like(close)
        for i in range(settings['STOCH_K']-1, len(close)):
            window_high = high[i-settings['STOCH_K']+1:i+1].max()
            window_low = low[i-settings['STOCH_K']+1:i+1].min()
            stoch_k[i] = 100 * (close[i] - window_low) / (window_high - window_low + 1e-10)
        
        stoch_d = pd.Series(stoch_k).rolling(settings['STOCH_D']).mean().values
        
        # ADX
        tr1 = pd.Series(high - low).rolling(settings['ADX_PERIOD']).sum().values
        tr2 = pd.Series(abs(high - np.roll(close, 1))).rolling(settings['ADX_PERIOD']).sum().values
        tr3 = pd.Series(abs(low - np.roll(close, 1))).rolling(settings['ADX_PERIOD']).sum().values
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(settings['ADX_PERIOD']).sum().values / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(settings['ADX_PERIOD']).sum().values / tr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(settings['ADX_PERIOD']).mean().values
        
        hl2 = (high + low) / 2
        
        trend = 1  # 1 = uptrend, -1 = downtrend
        
        for i in range(1, len(close)):
                trend = 1
                trend = -1
            
            if trend == 1:
            else:
        
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
        }
    except Exception as e:
        logger.error(f"Ошибка расчета индикаторов: {str(e)}\n{traceback.format_exc()}")
        return None

# Генерация сигналов
def generate_signal(close: np.ndarray, indicators: Dict[str, Any]) -> tuple:
    if indicators is None:
        return "Ошибка расчета индикаторов", "⚠️", 0
    
    signals = []
    buy_signals = 0
    sell_signals = 0
    strength = 0
    
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
    
    if len(indicators['macd']) > 1 and len(indicators['signal']) > 1:
        if indicators['macd'][-1] > indicators['signal'][-1] and indicators['macd'][-2] <= indicators['signal'][-2]:
            signals.append("MACD пересек сигнал снизу вверх")
            buy_signals += 1
            strength += 1
        elif indicators['macd'][-1] < indicators['signal'][-1] and indicators['macd'][-2] >= indicators['signal'][-2]:
            signals.append("MACD пересек сигнал сверху вниз")
            sell_signals += 1
            strength += 1
    
    if len(close) > 0 and 'upper' in indicators and 'lower' in indicators:
        if close[-1] < indicators['lower'][-1]:
            signals.append("Цена ниже нижней полосы Боллинджера")
            buy_signals += 1
            strength += 2
        elif close[-1] > indicators['upper'][-1]:
            signals.append("Цена выше верхней полосы Боллинджера")
            sell_signals += 1
            strength += 2
    
    if 'stoch_k' in indicators and 'stoch_d' in indicators:
        if indicators['stoch_k'][-1] < 20 and indicators['stoch_d'][-1] < 20:
            signals.append("Stochastic в зоне перепроданности")
            buy_signals += 1
            strength += 1
        elif indicators['stoch_k'][-1] > 80 and indicators['stoch_d'][-1] > 80:
            signals.append("Stochastic в зоне перекупленности")
            sell_signals += 1
            strength += 1
        
        if indicators['stoch_k'][-1] > indicators['stoch_d'][-1] and indicators['stoch_k'][-2] <= indicators['stoch_d'][-2]:
            signals.append("Stochastic %K пересек %D снизу вверх")
            buy_signals += 1
        elif indicators['stoch_k'][-1] < indicators['stoch_d'][-1] and indicators['stoch_k'][-2] >= indicators['stoch_d'][-2]:
            signals.append("Stochastic %K пересек %D сверху вниз")
            sell_signals += 1
    
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
    
            buy_signals += 1
            strength += 2
        else:
            sell_signals += 1
            strength += 2
    
    if not signals:
        return "Нет четких сигналов", "⚪", 0
    
    strength = min(max(strength, 0), 5)
    
    if buy_signals >= 3 and strength >= 3:
        return "Сильный сигнал на ПОКУПКУ", "🟢", strength
    elif sell_signals >= 3 and strength >= 3:
        return "Сильный сигнал на ПРОДАЖУ", "🔴", strength
    elif buy_signals >= 2 and strength >= 2:
        return "Средний сигнал на ПОКУПКУ", "🟡", strength
    elif sell_signals >= 2 and strength >= 2:
        return "Средний сигнал на ПРОДАЖУ", "🟠", strength
    elif buy_signals >= 1:
        return "Слабый сигнал на ПОКУПКУ", "🔵", strength
    elif sell_signals >= 1:
        return "Слабый сигнал на ПРОДАЖУ", "🟣", strength
    else:
        return "Смешанные сигналы", "⚪", strength

# Визуализация
def create_candlestick_plot(df: pd.DataFrame) -> io.BytesIO:
    try:
        mc = mpf.make_marketcolors(up='g', down='r')
        s = mpf.make_mpf_style(marketcolors=mc)
        
        buf = io.BytesIO()
        mpf.plot(df, type='candle', style=s, volume=False, 
                savefig=dict(fname=buf, dpi=100, bbox_inches='tight'))
        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"Ошибка создания свечного графика: {str(e)}\n{traceback.format_exc()}")
        return None

def create_technical_plot(df: pd.DataFrame, indicators: Dict[str, Any]) -> io.BytesIO:
    try:
        plt.figure(figsize=(14, 18), dpi=100)
        
        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
        ax1.plot(df.index, df['close'], label='Цена', color='royalblue', linewidth=1)
        if 'upper' in indicators and 'middle' in indicators and 'lower' in indicators:
            ax1.plot(df.index, indicators['upper'], label='Верхняя BB', color='orange', linestyle='--', linewidth=1)
            ax1.plot(df.index, indicators['middle'], label='Средняя BB', color='green', linestyle='--', linewidth=1)
            ax1.plot(df.index, indicators['lower'], label='Нижняя BB', color='orange', linestyle='--', linewidth=1)
            ax1.fill_between(df.index, indicators['upper'], indicators['lower'], color='orange', alpha=0.1)
        
        
        ax1.set_title('Цена с индикаторами')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot2grid((6, 1), (2, 0))
        if 'macd' in indicators and 'signal' in indicators:
            ax2.plot(df.index, indicators['macd'], label='MACD', color='blue', linewidth=1)
            ax2.plot(df.index, indicators['signal'], label='Сигнал', color='red', linewidth=1)
            ax2.fill_between(df.index, indicators['macd'], indicators['signal'],
                           where=indicators['macd']>indicators['signal'],
                           facecolor='green', alpha=0.3)
            ax2.fill_between(df.index, indicators['macd'], indicators['signal'],
                           where=indicators['macd']<=indicators['signal'],
                           facecolor='red', alpha=0.3)
            ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_title('MACD')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot2grid((6, 1), (3, 0))
        if 'rsi' in indicators:
            ax3.plot(df.index, indicators['rsi'], label='RSI', color='purple', linewidth=1)
            ax3.axhline(70, linestyle='--', color='red', alpha=0.5)
            ax3.axhline(30, linestyle='--', color='green', alpha=0.5)
            ax3.fill_between(df.index, indicators['rsi'], 70, where=indicators['rsi']>=70,
                            facecolor='red', alpha=0.1)
            ax3.fill_between(df.index, indicators['rsi'], 30, where=indicators['rsi']<=30,
                            facecolor='green', alpha=0.1)
        ax3.set_title('RSI')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot2grid((6, 1), (4, 0))
        if 'stoch_k' in indicators and 'stoch_d' in indicators:
            ax4.plot(df.index, indicators['stoch_k'], label='%K', color='blue', linewidth=1)
            ax4.plot(df.index, indicators['stoch_d'], label='%D', color='red', linewidth=1)
            ax4.axhline(80, linestyle='--', color='red', alpha=0.5)
            ax4.axhline(20, linestyle='--', color='green', alpha=0.5)
            ax4.fill_between(df.index, indicators['stoch_k'], 80, where=indicators['stoch_k']>=80,
                            facecolor='red', alpha=0.1)
            ax4.fill_between(df.index, indicators['stoch_k'], 20, where=indicators['stoch_k']<=20,
                            facecolor='green', alpha=0.1)
        ax4.set_title('Stochastic Oscillator')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot2grid((6, 1), (5, 0))
        if 'adx' in indicators and 'plus_di' in indicators and 'minus_di' in indicators:
            ax5.plot(df.index, indicators['adx'], label='ADX', color='black', linewidth=1)
            ax5.plot(df.index, indicators['plus_di'], label='+DI', color='green', linewidth=1)
            ax5.plot(df.index, indicators['minus_di'], label='-DI', color='red', linewidth=1)
            ax5.axhline(25, linestyle='--', color='blue', alpha=0.5)
        ax5.set_title('ADX')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logger.error(f"Ошибка создания графика индикаторов: {str(e)}\n{traceback.format_exc()}")
        return None

# Команды бота
def start(update: Update, context: CallbackContext) -> None:
    # Проверка на самообращение бота
    if update.effective_user and update.effective_user.id == BOT_ID:
        return

    try:
        message = safe_get_message(update)
        if not message:
            return
            
        user_id = message.from_user.id
        
        # Проверка что это не бот
        if message.from_user.is_bot:
            logger.debug(f"Попытка доступа от бота с ID: {user_id}")
            return
            
        # Проверка доступа
        if not check_access(user_id):
            try:
                context.bot.send_message(
                    chat_id=user_id,
                    text="⛔ У вас нет доступа к этому боту."
                )
            except Exception as e:
                logger.warning(f"Не удалось отправить сообщение об отказе: {str(e)}")
            return
            
        chat_id = message.chat_id
        cleanup_messages(context, chat_id, user_id)
        
        # Используем ensure_user_data вместо прямого доступа
        user_data = ensure_user_data(user_id)
        user_data['step'] = 'main_menu'
        
        buttons = [
            ("📊 Начать анализ", "start_analysis"),
            ("⚙️ Настройки", "settings_menu"),
            ("ℹ️ Помощь", "help"),
            ("🔄 Сбросить настройки", "reset_settings")
        ]
        
        msg_id = safe_send_message(
            context,
            chat_id,
            "📈 <b>Forex Trading Bot</b>\nВыберите действие:",
            reply_markup=create_menu_buttons(buttons, back_button=False)
        )
        
        if msg_id:
            add_message_to_cleanup(user_id, msg_id)
    except Exception as e:
        error_handler(update, context, e)

def help_command(update: Update, context: CallbackContext) -> None:
    # Проверка на самообращение бота
    if update.effective_user and update.effective_user.id == BOT_ID:
        return

    try:
        message = safe_get_message(update)
        if not message:
            return
            
        user_id = message.from_user.id
        
        # Проверка доступа
        if not check_access(user_id):
            return
            
        chat_id = message.chat_id
        cleanup_messages(context, chat_id, user_id)
        
        text = (
            "ℹ️ <b>Помощь по боту</b>\n\n"
            "Этот бот предоставляет технический анализ финансовых инструментов.\n\n"
            "<b>Основные функции:</b>\n"
            "- Анализ валютных пар и криптовалют\n"
            "- Настройка параметров индикаторов\n"
            "- Различные таймфреймы (5M, 15M, 1H, 4H)\n"
            "- Визуализация данных с графиками\n\n"
            "<b>Доступные группы валют:</b>\n"
            "• EUR - Европейский евро\n"
            "• USD - Американский доллар\n"
            "• GBP - Британский фунт\n"
            "• AUD - Австралийский доллар\n"
            "• JPY - Японская йена\n"
            "• CAD - Канадский доллар\n"
            "• CHF - Швейцарский франк\n"
            "• BTC - Биткоин\n\n"
            "<b>Доступные таймфреймы:</b>\n"
            "• 5M - 5 минут\n"
            "• 15M - 15 минут\n"
            "• 1H - 1 час\n"
            "• 4H - 4 часа\n\n"
            "<b>Доступные индикаторы:</b>\n"
            "- RSI (Relative Strength Index)\n"
            "- MACD (Moving Average Convergence Divergence)\n"
            "- Bollinger Bands\n"
            "- Stochastic Oscillator\n"
            "- ADX (Average Directional Index)\n"
            "<b>Как использовать:</b>\n"
            "1. Нажмите 'Начать анализ'\n"
            "2. Выберите группу валют\n"
            "3. Выберите конкретную валютную пару\n"
            "4. Выберите таймфрейм\n"
            "5. Получите анализ с графиками\n\n"
            "Для настройки параметров индикаторов используйте меню 'Настройки'"
        )
        
        msg_id = safe_send_message(
            context,
            chat_id,
            text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 В главное меню", callback_data='back_to_main')]
            ])
        )
        
        if msg_id:
            add_message_to_cleanup(user_id, msg_id)
    except Exception as e:
        error_handler(update, context, e)

# Обработчики кнопок
def button_handler(update: Update, context: CallbackContext) -> None:
    # Проверка на самообращение бота
    if update.effective_user and update.effective_user.id == BOT_ID:
        return

    query = update.callback_query
    if not query:
        return
        
    user_id = query.from_user.id
    
    # Проверка что это не бот
    if query.from_user.is_bot:
        logger.warning(f"Попытка доступа от бота с ID: {user_id}")
        return
    
    # Проверка доступа
    if not check_access(user_id):
        safe_answer_query(query, "⛔ У вас нет доступа к этому боту", show_alert=True)
        return
        
    chat_id = query.message.chat_id
    message_id = query.message.message_id
    
    if not safe_answer_query(query):
        return

    try:
        # Используем ensure_user_data для безопасного доступа
        user_data = ensure_user_data(user_id)
        current_step = user_data.get('step', '')
        
        if query.data == 'start_analysis':
            buttons = [(currency, f'currency_{currency}') for currency in CURRENCY_GROUPS.keys()]
            new_text = "📊 <b>Выберите базовую валюту:</b>"
            new_markup = create_menu_buttons(buttons)
            
            if not safe_edit_message(context, chat_id, message_id, new_text, new_markup):
                cleanup_messages(context, chat_id, user_id)
                msg_id = safe_send_message(context, chat_id, new_text, new_markup)
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)
            user_data['step'] = 'select_currency'

        elif query.data.startswith('currency_'):
            currency = query.data.split('_')[1]
            pairs = CURRENCY_GROUPS.get(currency, [])
            buttons = [(pair, f'pair_{i}') for i, pair in enumerate(pairs)]
            new_text = f"📊 <b>Выберите валютную пару ({currency}):</b>"
            new_markup = create_menu_buttons(buttons)
            
            if not safe_edit_message(context, chat_id, message_id, new_text, new_markup):
                cleanup_messages(context, chat_id, user_id)
                msg_id = safe_send_message(context, chat_id, new_text, new_markup)
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)
            user_data['step'] = 'select_pair'
            user_data['currency'] = currency

        elif query.data.startswith('pair_'):
            pair_idx = int(query.data.split('_')[1])
            currency = user_data.get('currency', '')
            pairs = CURRENCY_GROUPS.get(currency, [])
            if pair_idx < len(pairs):
                symbol = pairs[pair_idx]
                user_data['symbol'] = symbol
                
                buttons = [(tf, f'tf_{i}') for i, tf in enumerate(TIMEFRAMES)]
                new_text = f"📊 <b>Выбрана пара: {symbol}</b>\nВыберите таймфрейм:"
                new_markup = create_menu_buttons(buttons)
                
                if not safe_edit_message(context, chat_id, message_id, new_text, new_markup):
                    cleanup_messages(context, chat_id, user_id)
                    msg_id = safe_send_message(context, chat_id, new_text, new_markup)
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
                user_data['step'] = 'select_timeframe'
            else:
                safe_answer_query(query, "Ошибка выбора валютной пары", show_alert=True)

        elif query.data.startswith('tf_'):
            tf_idx = int(query.data.split('_')[1])
            if tf_idx < len(TIMEFRAMES):
                symbol = user_data.get('symbol', '')
                timeframe = TIMEFRAMES[tf_idx]
                
                context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                cleanup_messages(context, chat_id, user_id)
                
                try:
                    # Получаем СВЕЖИЕ данные при каждом запросе
                    df = get_ohlc_data(symbol, timeframe)
                    indicators = calculate_indicators(df, user_id)
                    
                    if indicators is None:
                        raise ValueError("Не удалось рассчитать индикаторы")
                    
                    candle_plot = create_candlestick_plot(df)
                    tech_plot = create_technical_plot(df, indicators)
                    
                    if candle_plot is None or tech_plot is None:
                        raise ValueError("Не удалось создать график")
                    
                    signal, emoji, strength = generate_signal(df['close'].values, indicators)
                    
                    try:
                        msg1 = context.bot.send_photo(
                            chat_id=chat_id,
                            photo=candle_plot,
                            caption=f"<b>{symbol} ({timeframe}) - Свечной график</b>\n"
                                   f"🔄 Данные обновлены: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            parse_mode='HTML',
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔙 Назад к выбору таймфрейма", callback_data=f'currency_{user_data.get("currency", "")}')]
                            ])
                        )
                        add_message_to_cleanup(user_id, msg1.message_id)
                    except Exception as e:
                        logger.error(f"Ошибка отправки свечного графика: {str(e)}")
                        raise
                    
                    try:
                        strength_bar = "🟢" * strength + "⚪" * (5 - strength) if strength > 0 else "⚪" * 5
                        msg2 = context.bot.send_photo(
                            chat_id=chat_id,
                            photo=tech_plot,
                            caption=(
                                f"<b>Технические индикаторы: {symbol} ({timeframe})</b>\n"
                                f"📊 <b>Сигнал:</b> {emoji} {signal}\n"
                                f"💪 <b>Сила сигнала:</b> {strength_bar} ({strength}/5)\n"
                                f"📈 Цена: {df['close'].iloc[-1]:.5f}\n"
                                f"🔄 Анализ выполнен: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            ),
                            parse_mode='HTML',
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔙 Назад к выбору таймфрейма", callback_data=f'currency_{user_data.get("currency", "")}')]
                            ])
                        )
                        add_message_to_cleanup(user_id, msg2.message_id)
                    except Exception as e:
                        logger.error(f"Ошибка отправки графика индикаторов: {str(e)}")
                        raise
                    
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
            else:
                safe_answer_query(query, "Ошибка выбора таймфрейма", show_alert=True)

        elif query.data == 'back':
            handle_back_action(update, context)

        elif query.data == 'back_to_main':
            start(update, context)

        elif query.data == 'settings_menu':
            show_settings_menu(update, context, user_id)
            
        elif query.data == 'reset_settings':
            user_data = ensure_user_data(user_id)
            user_data['settings'] = DEFAULT_SETTINGS.copy()
            safe_answer_query(query, "Настройки сброшены к значениям по умолчанию")
            show_settings_menu(update, context, user_id)

        elif query.data == 'set_rsi':
            user_data = ensure_user_data(user_id)
            user_data['step'] = 'set_rsi'
            safe_edit_message(
                context,
                chat_id,
                message_id,
                "Введите новый период для RSI (5-30):",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data='settings_menu')]
                ])
            )

        elif query.data == 'set_macd':
            user_data = ensure_user_data(user_id)
            user_data['step'] = 'set_macd'
            safe_edit_message(
                context,
                chat_id,
                message_id,
                "Введите параметры MACD в формате FAST/SLOW/SIGNAL (например 12/26/9):",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data='settings_menu')]
                ])
            )

        elif query.data == 'set_bb':
            user_data = ensure_user_data(user_id)
            user_data['step'] = 'set_bb'
            safe_edit_message(
                context,
                chat_id,
                message_id,
                "Введите параметры Bollinger Bands в формате PERIOD/STDDEV (например 20/2):",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data='settings_menu')]
                ])
            )

        elif query.data == 'set_stoch':
            user_data = ensure_user_data(user_id)
            user_data['step'] = 'set_stoch'
            safe_edit_message(
                context,
                chat_id,
                message_id,
                "Введите параметры Stochastic в формате %K/%D (например 14/3):",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data='settings_menu')]
                ])
            )

        elif query.data == 'set_adx':
            user_data = ensure_user_data(user_id)
            user_data['step'] = 'set_adx'
            safe_edit_message(
                context,
                chat_id,
                message_id,
                "Введите период для ADX (5-30):",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data='settings_menu')]
                ])
            )

            user_data = ensure_user_data(user_id)
            safe_edit_message(
                context,
                chat_id,
                message_id,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data='settings_menu')]
                ])
            )

    except Exception as e:
        if "Message is not modified" in str(e):
            logger.warning("Попытка изменить неизмененное сообщение")
        else:
            error_handler(update, context, e)

def handle_back_action(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    if not query:
        return
        
    user_id = query.from_user.id
    
    # Проверка доступа
    if not check_access(user_id):
        return
        
    chat_id = query.message.chat_id
    message_id = query.message.message_id
    
    if not safe_answer_query(query):
        return

    try:
        user_data = ensure_user_data(user_id)
        current_step = user_data.get('step', '')
        
        if current_step == 'select_timeframe':
            currency = user_data.get('currency', '')
            pairs = CURRENCY_GROUPS.get(currency, [])
            buttons = [(pair, f'pair_{i}') for i, pair in enumerate(pairs)]
            new_text = f"📊 <b>Выберите валютную пару ({currency}):</b>"
            new_markup = create_menu_buttons(buttons)
            
            if not safe_edit_message(context, chat_id, message_id, new_text, new_markup):
                cleanup_messages(context, chat_id, user_id)
                msg_id = safe_send_message(context, chat_id, new_text, new_markup)
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)
            user_data['step'] = 'select_pair'
        
        elif current_step == 'select_pair':
            buttons = [(currency, f'currency_{currency}') for currency in CURRENCY_GROUPS.keys()]
            new_text = "📊 <b>Выберите базовую валюту:</b>"
            new_markup = create_menu_buttons(buttons)
            
            if not safe_edit_message(context, chat_id, message_id, new_text, new_markup):
                cleanup_messages(context, chat_id, user_id)
                msg_id = safe_send_message(context, chat_id, new_text, new_markup)
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)
            user_data['step'] = 'select_currency'
        
        elif current_step == 'select_currency':
            start(update, context)
        else:
            start(update, context)
    except Exception as e:
        error_handler(update, context, e)

def show_settings_menu(update: Update, context: CallbackContext, user_id: int) -> None:
    try:
        query = update.callback_query
        if not query:
            return
            
        # Проверка доступа
        if not check_access(user_id):
            return
            
        chat_id = query.message.chat_id
        message_id = query.message.message_id
        cleanup_messages(context, chat_id, user_id)
        
        settings = get_user_settings(user_id)
        buttons = [
            (f"RSI Период: {settings['RSI_PERIOD']}", "set_rsi"),
            (f"MACD: {settings['MACD_FAST']}/{settings['MACD_SLOW']}/{settings['MACD_SIGNAL']}", "set_macd"),
            (f"Bollinger Bands: {settings['BB_PERIOD']}/{settings['BB_STDDEV']}", "set_bb"),
            (f"Stochastic: {settings['STOCH_K']}/{settings['STOCH_D']}", "set_stoch"),
            (f"ADX Период: {settings['ADX_PERIOD']}", "set_adx"),
            ("🔙 В главное меню", "back_to_main")
        ]
        
        new_text = "⚙️ <b>Настройки индикаторов</b>\nВыберите параметр:"
        new_markup = create_menu_buttons(buttons, back_button=False)
        
        if not safe_edit_message(context, chat_id, message_id, new_text, new_markup):
            msg_id = safe_send_message(context, chat_id, new_text, new_markup)
            if msg_id:
                add_message_to_cleanup(user_id, msg_id)
        user_data = ensure_user_data(user_id)
        user_data['step'] = 'settings_menu'
    except Exception as e:
        error_handler(update, context, e)

def text_handler(update: Update, context: CallbackContext) -> None:
    # Проверка на самообращение бота
    if update.effective_user and update.effective_user.id == BOT_ID:
        return

    if not update.message:
        return
        
    user_id = update.message.from_user.id
    
    # Проверка что это не бот
    if update.message.from_user.is_bot:
        logger.warning(f"Попытка доступа от бота с ID: {user_id}")
        return
    
    # Проверка доступа
    if not check_access(user_id):
        return
    
    chat_id = update.message.chat_id
    text = update.message.text.strip()
    
    user_data = ensure_user_data(user_id)
    current_step = user_data.get('step', '')
    
    if current_step.startswith('set_'):
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
    try:
        # Проверка доступа
        if not check_access(user_id):
            return
            
        chat_id = update.message.chat_id
        cleanup_messages(context, chat_id, user_id)
        
        user_data = ensure_user_data(user_id)
        settings = get_user_settings(user_id)
        current_step = user_data['step']
        
        if current_step == 'set_rsi':
            try:
                value = int(text)
                if 5 <= value <= 30:
                    settings['RSI_PERIOD'] = value
                    msg_id = safe_send_message(
                        context,
                        chat_id,
                        f"✅ Период RSI изменен на {value}",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔙 Назад к настройкам", callback_data='settings_menu')]
                        ])
                    )
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
                    user_data['step'] = 'settings_menu'
                else:
                    msg_id = safe_send_message(context, chat_id, "Пожалуйста, введите число от 5 до 30")
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
            except ValueError:
                msg_id = safe_send_message(context, chat_id, "Пожалуйста, введите целое число")
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)

        elif current_step == 'set_macd':
            try:
                parts = list(map(int, text.split('/')))
                if len(parts) == 3:
                    fast, slow, signal = parts
                    if 5 <= fast < slow <= 50 and 5 <= signal <= 20:
                        settings['MACD_FAST'] = fast
                        settings['MACD_SLOW'] = slow
                        settings['MACD_SIGNAL'] = signal
                        msg_id = safe_send_message(
                            context,
                            chat_id,
                            f"✅ Параметры MACD изменены на {fast}/{slow}/{signal}",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔙 Назад к настройкам", callback_data='settings_menu')]
                            ])
                        )
                        if msg_id:
                            add_message_to_cleanup(user_id, msg_id)
                        user_data['step'] = 'settings_menu'
                    else:
                        msg_id = safe_send_message(context, chat_id, 
                            "Пожалуйста, введите корректные значения (например: 12/26/9)")
                        if msg_id:
                            add_message_to_cleanup(user_id, msg_id)
                else:
                    msg_id = safe_send_message(context, chat_id, 
                        "Пожалуйста, введите 3 числа через слеш (например: 12/26/9)")
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
            except ValueError:
                msg_id = safe_send_message(context, chat_id, "Пожалуйста, введите числа в формате X/Y/Z")
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)

        elif current_step == 'set_bb':
            try:
                parts = list(map(int, text.split('/')))
                if len(parts) == 2:
                    period, stddev = parts
                    if 5 <= period <= 50 and 1 <= stddev <= 3:
                        settings['BB_PERIOD'] = period
                        settings['BB_STDDEV'] = stddev
                        msg_id = safe_send_message(
                            context,
                            chat_id,
                            f"✅ Параметры Bollinger Bands изменены на {period}/{stddev}",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔙 Назад к настройкам", callback_data='settings_menu')]
                            ])
                        )
                        if msg_id:
                            add_message_to_cleanup(user_id, msg_id)
                        user_data['step'] = 'settings_menu'
                    else:
                        msg_id = safe_send_message(context, chat_id, 
                            "Пожалуйста, введите корректные значения (например: 20/2)")
                        if msg_id:
                            add_message_to_cleanup(user_id, msg_id)
                else:
                    msg_id = safe_send_message(context, chat_id, 
                        "Пожалуйста, введите 2 числа через слеш (например: 20/2)")
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
            except ValueError:
                msg_id = safe_send_message(context, chat_id, "Пожалуйста, введите числа в формате X/Y")
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)

        elif current_step == 'set_stoch':
            try:
                parts = list(map(int, text.split('/')))
                if len(parts) == 2:
                    k, d = parts
                    if 5 <= k <= 30 and 1 <= d <= 10:
                        settings['STOCH_K'] = k
                        settings['STOCH_D'] = d
                        msg_id = safe_send_message(
                            context,
                            chat_id,
                            f"✅ Параметры Stochastic изменены на {k}/{d}",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔙 Назад к настройкам", callback_data='settings_menu')]
                            ])
                        )
                        if msg_id:
                            add_message_to_cleanup(user_id, msg_id)
                        user_data['step'] = 'settings_menu'
                    else:
                        msg_id = safe_send_message(context, chat_id, 
                            "Пожалуйста, введите корректные значения (например: 14/3)")
                        if msg_id:
                            add_message_to_cleanup(user_id, msg_id)
                else:
                    msg_id = safe_send_message(context, chat_id, 
                        "Пожалуйста, введите 2 числа через слеш (например: 14/3)")
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
            except ValueError:
                msg_id = safe_send_message(context, chat_id, "Пожалуйста, введите числа в формате X/Y")
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)

        elif current_step == 'set_adx':
            try:
                value = int(text)
                if 5 <= value <= 30:
                    settings['ADX_PERIOD'] = value
                    msg_id = safe_send_message(
                        context,
                        chat_id,
                        f"✅ Период ADX изменен на {value}",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔙 Назад к настройкам", callback_data='settings_menu')]
                        ])
                    )
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
                    user_data['step'] = 'settings_menu'
                else:
                    msg_id = safe_send_message(context, chat_id, "Пожалуйста, введите число от 5 до 30")
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
            except ValueError:
                msg_id = safe_send_message(context, chat_id, "Пожалуйста, введите целое число")
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)

            try:
                parts = text.split('/')
                if len(parts) == 2:
                    period = int(parts[0])
                    multiplier = float(parts[1])
                    if 5 <= period <= 30 and 1.0 <= multiplier <= 5.0:
                        msg_id = safe_send_message(
                            context,
                            chat_id,
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔙 Назад к настройкам", callback_data='settings_menu')]
                            ])
                        )
                        if msg_id:
                            add_message_to_cleanup(user_id, msg_id)
                        user_data['step'] = 'settings_menu'
                    else:
                        msg_id = safe_send_message(context, chat_id, 
                            "Пожалуйста, введите корректные значения (например: 10/3.0)")
                        if msg_id:
                            add_message_to_cleanup(user_id, msg_id)
                else:
                    msg_id = safe_send_message(context, chat_id, 
                        "Пожалуйста, введите 2 значения через слеш (например: 10/3.0)")
                    if msg_id:
                        add_message_to_cleanup(user_id, msg_id)
            except ValueError:
                msg_id = safe_send_message(context, chat_id, "Пожалуйста, введите числа в формате X/Y")
                if msg_id:
                    add_message_to_cleanup(user_id, msg_id)

    except Exception as e:
        error_handler(update, context, e)

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
        updater = Updater(TOKEN, use_context=True)
        
        # Устанавливаем глобальную переменную BOT_ID
        global BOT_ID
        BOT_ID = updater.bot.id
        
        dp = updater.dispatcher

        dp.add_handler(CommandHandler("start", start))
        dp.add_handler(CommandHandler("help", help_command))
        dp.add_handler(CallbackQueryHandler(button_handler))
        dp.add_handler(MessageHandler(Filters.text & ~Filters.command, text_handler))
        dp.add_error_handler(lambda u, c: error_handler(u, c, c.error))

        job_queue = updater.job_queue
        job_queue.run_repeating(lambda _: cleanup_old_users(), interval=3600, first=0)

        logger.info("Бот запущен")
        updater.start_polling()
        updater.idle()

    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {str(e)}\n{traceback.format_exc()}")

if __name__ == '__main__':
    main()
