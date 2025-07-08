import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Optional
from bot.config import TWELVE_DATA_KEY

def get_ohlc_data(symbol: str = "EUR/USD", timeframe: str = "1H") -> pd.DataFrame:
    """
    Получение OHLC данных от API или генерация тестовых данных при ошибке.
    
    Args:
        symbol: Торговый инструмент (например, "EUR/USD")
        timeframe: Таймфрейм ("5M", "15M", "1H", "4H")
        
    Returns:
        DataFrame с колонками ['open', 'high', 'low', 'close']
    """
    try:
        interval_map = {'5M': '5min', '15M': '15min', '1H': '1h', '4H': '4h'}
        params = {
            'symbol': symbol,
            'interval': interval_map.get(timeframe, '1h'),
            'apikey': TWELVE_DATA_KEY,
            'outputsize': 500,
            'format': 'JSON'
        }
        
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
            
        return df[['open', 'high', 'low', 'close']]
        
    except Exception as e:
        print(f"Ошибка получения данных: {str(e)}")
        return generate_test_data(symbol, timeframe)

def generate_test_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Генерация тестовых данных при недоступности API.
    
    Args:
        symbol: Торговый инструмент
        timeframe: Таймфрейм
        
    Returns:
        DataFrame с тестовыми данными
    """
    print("Генерация тестовых данных")
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