from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from bot.config import CURRENCY_GROUPS, TIMEFRAMES, DEFAULT_SETTINGS

def create_menu_buttons(buttons_data: list, back_button: bool = True) -> InlineKeyboardMarkup:
    """
    Создание клавиатуры меню из списка кнопок.
    
    Args:
        buttons_data: Список кортежей (текст, callback_data)
        back_button: Добавлять ли кнопку "Назад"
    
    Returns:
        InlineKeyboardMarkup: Объект клавиатуры
    """
    keyboard = [[InlineKeyboardButton(text, callback_data=callback)] for text, callback in buttons_data]
    if back_button:
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data='back')])
    return InlineKeyboardMarkup(keyboard)

def create_settings_keyboard(settings: dict) -> InlineKeyboardMarkup:
    """
    Создание клавиатуры настроек индикаторов.
    
    Args:
        settings: Словарь с текущими настройками
        
    Returns:
        InlineKeyboardMarkup: Клавиатура настроек
    """
    keyboard = [
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
    return InlineKeyboardMarkup(keyboard)

def create_currency_keyboard() -> InlineKeyboardMarkup:
    """
    Создание клавиатуры выбора базовой валюты.
    
    Returns:
        InlineKeyboardMarkup: Клавиатура с валютами
    """
    buttons = [
        ("🇪🇺 EUR", 'currency_EUR'),
        ("🇺🇸 USD", 'currency_USD'),
        ("🇬🇧 GBP", 'currency_GBP'),
        ("🇦🇺 AUD", 'currency_AUD'),
        ("🇯🇵 JPY", 'currency_JPY'),
        ("🇨🇦 CAD", 'currency_CAD'),
        ("🇨🇭 CHF", 'currency_CHF'),
        ("₿ BTC", 'currency_BTC')
    ]
    return InlineKeyboardMarkup([[InlineKeyboardButton(text, callback_data=data)] for text, data in buttons])

def create_pairs_keyboard(currency: str) -> InlineKeyboardMarkup:
    """
    Создание клавиатуры выбора валютных пар для выбранной валюты.
    
    Args:
        currency: Код базовой валюты (например, 'EUR')
        
    Returns:
        InlineKeyboardMarkup: Клавиатура с парами
    """
    pairs = CURRENCY_GROUPS.get(currency, [])
    buttons = [(pair, f'pair_{i}') for i, pair in enumerate(pairs)]
    keyboard = [[InlineKeyboardButton(text, callback_data=data)] for text, data in buttons]
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data='back')])
    return InlineKeyboardMarkup(keyboard)

def create_timeframes_keyboard() -> InlineKeyboardMarkup:
    """
    Создание клавиатуры выбора таймфрейма.
    
    Returns:
        InlineKeyboardMarkup: Клавиатура с таймфреймами
    """
    buttons = [(tf, f'tf_{i}') for i, tf in enumerate(TIMEFRAMES)]
    keyboard = [[InlineKeyboardButton(text, callback_data=data)] for text, data in buttons]
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data='back')])
    return InlineKeyboardMarkup(keyboard)

def create_back_keyboard(back_data: str = 'back') -> InlineKeyboardMarkup:
    """
    Создание простой клавиатуры с кнопкой "Назад".
    
    Args:
        back_data: Callback_data для кнопки "Назад"
        
    Returns:
        InlineKeyboardMarkup: Клавиатура с одной кнопкой
    """
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data=back_data)]])