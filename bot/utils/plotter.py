import io
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from bot.config import config.COLORS

matplotlib.use('Agg')

def create_candlestick_plot(df: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Создание свечного графика с индикаторами."""
    try:
        # Настройка цветов
        mc = mpf.make_marketcolors(
            up=COLORS['up'],
            down=COLORS['down'],
            wick={'up':COLORS['wick_up'], 'down':COLORS['wick_down']},
            edge={'up':COLORS['wick_up'], 'down':COLORS['wick_down']}
        )
        
        # Стиль графика
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            gridcolor=COLORS['grid'],
            facecolor=COLORS['background'],
            edgecolor='#dddddd',
            figcolor='white'
        )
        
        # Добавление индикаторов
        add_plot = [
            mpf.make_addplot(indicators['ema50'], color=COLORS['ema_fast'], width=1.5),
            mpf.make_addplot(indicators['ema200'], color=COLORS['ema_slow'], width=1.5),
            mpf.make_addplot(indicators['supertrend'], color=COLORS['supertrend'], width=1)
        ]
        
        # Создание графика
        buf = io.BytesIO()
        fig, _ = mpf.plot(
            df,
            type='candle',
            style=s,
            addplot=add_plot,
            figscale=1.1,
            figratio=(10, 6),
            title="\n\n",
            returnfig=True
        )
        
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logging.error(f"Ошибка создания свечного графика: {str(e)}")
        return None

def create_enhanced_plot(df: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Создание расширенного графика с индикаторами."""
    try:
        # Настройка стиля
        plt.rcParams.update({
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            'axes.edgecolor': '#dddddd',
            'axes.linewidth': 0.8,
            'figure.facecolor': 'white'
        })
        
        # Создание фигуры
        plt.figure(figsize=(14, 22), dpi=120)
        
        # 1. График цены с индикаторами
        ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=2)
        ax1.plot(df.index, df['close'], label='Цена', color=COLORS['text'], linewidth=1.5)
        ax1.plot(df.index, indicators['ema50'], label='EMA50', color=COLORS['ema_fast'], linestyle='--', alpha=0.8)
        ax1.plot(df.index, indicators['ema200'], label='EMA200', color=COLORS['ema_slow'], linestyle='--', alpha=0.8)
        
        # Облако Ichimoku
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
        logging.error(f"Ошибка создания графика индикаторов: {str(e)}")
        return None
