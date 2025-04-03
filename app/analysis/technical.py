"""
Technical analysis module for Bitcoin Trading Bot
"""
import pandas as pd
import numpy as np
from app.config import logger

class TechnicalAnalyzer:
    """Class to perform technical analysis on price data"""
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Price rate of change
        df['roc'] = df['close'].pct_change(periods=10) * 100
        
        return df
    
    @staticmethod
    def detect_breakouts(df: pd.DataFrame, lookback_period: int = 20) -> pd.DataFrame:
        """
        Detect potential breakouts based on volatility and price action
        
        Args:
            df: DataFrame with OHLCV data and indicators
            lookback_period: Period to look back for resistance/support
            
        Returns:
            DataFrame with breakout signals
        """
        df = df.copy()
        
        # Calculate local highs and lows
        df['local_high'] = df['high'].rolling(window=lookback_period, center=False).max()
        df['local_low'] = df['low'].rolling(window=lookback_period, center=False).min()
        
        # Check if current candle breaks the local high/low
        df['breaks_high'] = df['close'] > df['local_high'].shift(1)
        df['breaks_low'] = df['close'] < df['local_low'].shift(1)
        
        # Check for increased volume
        df['volume_surge'] = df['volume'] > df['volume_sma_20'] * 1.5
        
        # Check for volatility expansion
        df['vol_expansion'] = df['atr'] > df['atr'].shift(1) * 1.2
        
        # Define breakout signals
        df['long_signal'] = (df['breaks_high'] & df['volume_surge'] & df['vol_expansion']).astype(int)
        df['short_signal'] = (df['breaks_low'] & df['volume_surge'] & df['vol_expansion']).astype(int)
        
        return df
    
    @staticmethod
    def prepare_data_for_llm(df: pd.DataFrame, rows: int = 50) -> str:
        """
        Prepare a subset of data for LLM analysis in CSV format
        
        Args:
            df: DataFrame with OHLCV data and indicators
            rows: Number of recent rows to include
            
        Returns:
            CSV string of the subset data
        """
        # Select relevant columns for analysis
        relevant_columns = [
            'open', 'high', 'low', 'close', 'volume', 
            'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 
            'bb_upper', 'bb_lower', 'atr', 'long_signal', 'short_signal'
        ]
        
        # Get the most recent rows
        subset = df.tail(rows)[relevant_columns]
        
        # Convert to CSV string
        csv_string = subset.to_csv()
        return csv_string