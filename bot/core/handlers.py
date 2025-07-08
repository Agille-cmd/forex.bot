import logging
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatAction
from telegram.ext import CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, Filters

from bot.config import config.TOKEN, config.ALLOWED_USER_IDS, config.DEFAULT_SETTINGS, config.CURRENCY_GROUPS, config.TIMEFRAMES, config.MULTI_TIMEFRAMES, config.COLORS
from bot.services.technical import TechnicalAnalyzer
from bot.services.data_fetcher import get_ohlc_data, generate_test_data
from bot.utils.plotter import create_candlestick_plot, create_enhanced_plot
from bot.utils.helpers import (
    check_access, ensure_user_data, safe_answer_query, safe_edit_message,
    safe_delete_message, safe_send_message, add_message_to_cleanup,
    cleanup_messages
)
from bot.core.keyboards import (
    create_menu_buttons, create_settings_keyboard,
    create_currency_keyboard, create_pairs_keyboard,
    create_timeframes_keyboard
)

# Глобальные переменные состояния
user_data: Dict[int, Dict[str, Any]] = {}
user_messages: Dict[int, List[int]] = {}
tech_analyzer = TechnicalAnalyzer()

def setup_handlers(app):
    """Настройка всех обработчиков для бота."""
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(Filters.text & ~Filters.command, text_handler))
    app.add_error_handler(error_handler)

def start(update: Update, context: CallbackContext) -> None:
    """Обработчик команды /start."""
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
    """Обработчик команды /help."""
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
    """Обработчик нажатий на кнопки."""
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
        
        # Оптимизация параметров
        optimized_settings = tech_analyzer.optimize_parameters(symbol, timeframe, user_id)
        user_data['settings'] = optimized_settings
        
        df = get_ohlc_data(symbol, timeframe)
        indicators = tech_analyzer.calculate_indicators(df, user_data['settings'])
        
        if indicators is None:
            raise ValueError("Не удалось рассчитать индикаторы")
        
        signal, emoji, strength = tech_analyzer.generate_signal(df['close'].values, indicators)
        
        # Мультитаймфреймное подтверждение
        if "BUY" in signal:
            if not tech_analyzer.check_multi_timeframe_confirmation(symbol, user_id, "BUY"):
                signal = "Неподтвержденный " + signal
                strength = max(0, strength - 1)
        elif "SELL" in signal:
            if not tech_analyzer.check_multi_timeframe_confirmation(symbol, user_id, "SELL"):
                signal = "Неподтвержденный " + signal
                strength = max(0, strength - 1)
        
        candle_plot = create_candlestick_plot(df, indicators)
        tech_plot = create_enhanced_plot(df, indicators)
        
        if candle_plot is None or tech_plot is None:
            raise ValueError("Не удалось создать график")
        
        send_analysis_results(context, chat_id, user_id, user_data, df, timeframe, candle_plot, tech_plot, signal, emoji, strength)
            
    except Exception as e:
        logging.error(f"Ошибка анализа: {str(e)}\n{traceback.format_exc()}")
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

def send_analysis_results(context, chat_id, user_id, user_data, df, timeframe, candle_plot, tech_plot, signal, emoji, strength):
    text = f"""
📊 <b>{user_data['symbol']} | {timeframe}</b>
━━━━━━━━━━━━━━━━
🔼 <b>Цена:</b> {df['close'].iloc[-1]:.5f}
🔽 <b>Изменение (24ч):</b> {((df['close'].iloc[-1] - df['close'].iloc[-24])/df['close'].iloc[-24]*100):+.2f}%
━━━━━━━━━━━━━━━━
🚀 <b>Сигнал:</b> {emoji} {signal}
"""
    
    try:
        msg1 = context.bot.send_photo(
            chat_id=chat_id,
            photo=candle_plot,
            caption=f"<b>{user_data['symbol']} ({timeframe}) - Свечной график</b>",
            parse_mode='HTML'
        )
        add_message_to_cleanup(user_id, msg1.message_id)
    except Exception as e:
        logging.error(f"Ошибка отправки свечного графика: {str(e)}")
        raise
    
    try:
        msg2 = context.bot.send_photo(
            chat_id=chat_id,
            photo=tech_plot,
            caption=text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🔄 Обновить", callback_data=f'tf_{TIMEFRAMES.index(timeframe)}'),
                    InlineKeyboardButton("🔙 Назад", callback_data=f'currency_{user_data.get("currency", "")}')
                ]
            ])
        )
        add_message_to_cleanup(user_id, msg2.message_id)
    except Exception as e:
        logging.error(f"Ошибка отправки графика индикаторов: {str(e)}")
        raise

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
    """Обработчик текстовых сообщений."""
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
    """Обработчик ошибок."""
    logging.error(f"Ошибка: {str(error)}\n{traceback.format_exc()}")
    
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
        logging.error(f"Ошибка в обработчике ошибок: {str(e)}")

def cleanup_and_send(context: CallbackContext, chat_id: int, user_id: int, text: str, buttons_data: list):
    """Очистка сообщений и отправка нового."""
    cleanup_messages(context, chat_id, user_id)
    msg_id = safe_send_message(
        context,
        chat_id,
        text,
        reply_markup=create_menu_buttons(buttons_data)
    )
    if msg_id:
        add_message_to_cleanup(user_id, msg_id)
