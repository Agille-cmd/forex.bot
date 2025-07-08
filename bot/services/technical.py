import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from bot.config import CANDLE_PATTERNS, DEFAULT_SETTINGS

class TechnicalAnalyzer:
    def __init__(self):
        pass

    def calculate_indicators(self, df: pd.DataFrame, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Расчет всех технических индикаторов для DataFrame с ценами.
        
        Args:
            df: DataFrame с колонками ['open', 'high', 'low', 'close']
            settings: Словарь с настройками индикаторов
            
        Returns:
            Словарь с рассчитанными индикаторами или None в случае ошибки
        """
        try:
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
                
                for name, pattern_func in CANDLE_PATTERNS.items():
                    if name in ['hammer', 'inverse_hammer']:
                        if pattern_func(o3, h3, l3, c3):
                            patterns.append(name)
                    elif name in ['bullish_engulfing', 'bearish_engulfing']:
                        if pattern_func(o2, h2, l2, c2, o3, h3, l3, c3):
                            patterns.append(name)
                    elif name == 'morning_star':
                        if pattern_func(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
                            patterns.append(name)
            
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
            print(f"Ошибка расчета индикаторов: {str(e)}")
            return None

    def generate_signal(self, close: np.ndarray, indicators: Dict[str, Any]) -> Tuple[str, str, int]:
        """
        Генерация торгового сигнала на основе индикаторов.
        
        Args:
            close: Массив цен закрытия
            indicators: Словарь с рассчитанными индикаторами
            
        Returns:
            Кортеж: (текст сигнала, эмодзи, сила сигнала)
        """
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
        if not self.is_good_time_to_trade():
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

    def is_good_time_to_trade(self) -> bool:
        """Проверка, хорошее ли время для торговли (по часам)."""
        hour = datetime.now().hour
        return (8 <= hour <= 12) or (14 <= hour <= 20)

    def check_multi_timeframe_confirmation(self, symbol: str, user_id: int, direction: str) -> bool:
        """
        Проверка подтверждения сигнала на нескольких таймфреймах.
        
        Args:
            symbol: Торговый инструмент
            user_id: ID пользователя
            direction: Направление ('BUY' или 'SELL')
            
        Returns:
            bool: Есть ли подтверждение
        """
        from bot.services.data_fetcher import get_ohlc_data
        from bot.core.handlers import user_data
        
        confirmations = 0
        user_settings = user_data.get(user_id, {}).get('settings', DEFAULT_SETTINGS)
        
        for timeframe in MULTI_TIMEFRAMES:
            df = get_ohlc_data(symbol, timeframe)
            indicators = self.calculate_indicators(df, user_settings)
            if indicators:
                signal, _, _ = self.generate_signal(df['close'].values, indicators)
                if direction == "BUY" and "BUY" in signal:
                    confirmations += 1
                elif direction == "SELL" and "SELL" in signal:
                    confirmations += 1
        
        return confirmations >= 2

    def optimize_parameters(self, symbol: str, timeframe: str, user_id: int) -> Dict[str, Any]:
        """
        Оптимизация параметров индикаторов для конкретной пары и таймфрейма.
        
        Args:
            symbol: Торговый инструмент
            timeframe: Таймфрейм
            user_id: ID пользователя
            
        Returns:
            Оптимизированные настройки
        """
        from bot.core.handlers import user_data
        from bot.services.data_fetcher import get_ohlc_data
        
        if user_id not in user_data:
            user_data[user_id] = {'optimized_params': {}}
        
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in user_data[user_id]['optimized_params']:
            return user_data[user_id]['optimized_params'][cache_key]
        
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
                    indicators = self.calculate_indicators(df, current_settings)
                    signals = []
                    for i in range(100, len(df)):
                        signal, _, strength = self.generate_signal(df['close'].values[:i], indicators)
                        if "BUY" in signal or "SELL" in signal:
                            signals.append((signal, strength, df['close'].values[i]))
                    
                    # Оцениваем качество сигналов
                    if len(signals) > 5:
                        score = sum(s[1] for s in signals) / len(signals)
                        if score > best_score:
                            best_score = score
                            best_settings = current_settings
        
        user_data[user_id]['optimized_params'][cache_key] = best_settings
        return best_settings