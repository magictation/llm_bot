"""
Enhanced reversal detection for the trading bot
Add to app/analysis/technical.py or create as app/analysis/reversal_detection.py
"""
from typing import Tuple
import pandas as pd
import numpy as np
from app.config import logger

class ReversalDetector:
    """Class to detect potential reversals and overextended moves"""
    
    @staticmethod
    def add_reversal_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add reversal detection indicators to DataFrame"""
        df = df.copy()
        
        try:
            # 1. Price change indicators
            # Calculate recent price changes over multiple time windows
            for window in [3, 5, 8, 13]:
                df[f'price_change_{window}'] = df['close'].pct_change(window) * 100
            
            # 2. Detect overextended moves
            # A significant move in one direction that may be due for a reversal
            df['overextended_up'] = (df['price_change_5'] > 10) & (df['price_change_3'] < 0)
            df['overextended_down'] = (df['price_change_5'] < -10) & (df['price_change_3'] > 0)
            
            # 3. Detect swing points using zigzag-like algorithm
            # Initialize columns
            df['swing_high'] = False
            df['swing_low'] = False
            
            # Detect swing points with 5-bar lookback
            for i in range(5, len(df) - 5):
                # Check for swing high
                if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, 4)) and \
                   all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, 4)):
                    df.loc[df.index[i], 'swing_high'] = True
                
                # Check for swing low
                if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, 4)) and \
                   all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, 4)):
                    df.loc[df.index[i], 'swing_low'] = True
            
            # 4. RSI divergence
            if 'rsi' in df.columns:
                # Initialize divergence columns
                df['bullish_divergence'] = False
                df['bearish_divergence'] = False
                
                # Look for bullish divergence (price making lower lows, RSI making higher lows)
                for i in range(10, len(df)):
                    if df['swing_low'].iloc[i]:
                        # Look back for previous swing low
                        for j in range(i-5, i-30, -1):
                            if j > 0 and df['swing_low'].iloc[j]:
                                # Check for bullish divergence
                                if df['low'].iloc[i] < df['low'].iloc[j] and \
                                   df['rsi'].iloc[i] > df['rsi'].iloc[j]:
                                    df.loc[df.index[i], 'bullish_divergence'] = True
                                break
                
                # Look for bearish divergence (price making higher highs, RSI making lower highs)
                for i in range(10, len(df)):
                    if df['swing_high'].iloc[i]:
                        # Look back for previous swing high
                        for j in range(i-5, i-30, -1):
                            if j > 0 and df['swing_high'].iloc[j]:
                                # Check for bearish divergence
                                if df['high'].iloc[i] > df['high'].iloc[j] and \
                                   df['rsi'].iloc[i] < df['rsi'].iloc[j]:
                                    df.loc[df.index[i], 'bearish_divergence'] = True
                                break
            
            # 5. Volume confirmation
            if 'volume' in df.columns:
                # Volume moving average
                df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
                
                # Check for declining volume on continued trend (sign of weakening momentum)
                df['declining_up_volume'] = (df['close'] > df['close'].shift(1)) & \
                                           (df['close'].shift(1) > df['close'].shift(2)) & \
                                           (df['volume'] < df['volume'].shift(1)) & \
                                           (df['volume'].shift(1) < df['volume'].shift(2))
                                           
                df['declining_down_volume'] = (df['close'] < df['close'].shift(1)) & \
                                             (df['close'].shift(1) < df['close'].shift(2)) & \
                                             (df['volume'] < df['volume'].shift(1)) & \
                                             (df['volume'].shift(1) < df['volume'].shift(2))
            
            # 6. Momentum slowdown detection
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                # Look for momentum slowdown in both directions
                df['momentum_weakening_up'] = (df['macd'] > 0) & (df['macd'] < df['macd'].shift(1)) & \
                                             (df['macd'].shift(1) < df['macd'].shift(2))
                                             
                df['momentum_weakening_down'] = (df['macd'] < 0) & (df['macd'] > df['macd'].shift(1)) & \
                                               (df['macd'].shift(1) > df['macd'].shift(2))
            
            # 7. Candlestick pattern detection (basic)
            # Doji (indecision)
            df['doji'] = abs(df['close'] - df['open']) < (0.1 * (df['high'] - df['low']))
            
            # Hammer (potential bullish reversal)
            df['hammer'] = (df['close'] > df['open']) & \
                          ((df['high'] - df['close']) < 0.2 * (df['high'] - df['low'])) & \
                          ((df['open'] - df['low']) > 2 * (df['high'] - df['open']))
            
            # Shooting star (potential bearish reversal)
            df['shooting_star'] = (df['close'] < df['open']) & \
                                 ((df['close'] - df['low']) < 0.2 * (df['high'] - df['low'])) & \
                                 ((df['high'] - df['open']) > 2 * (df['open'] - df['close']))
            
            # 8. Create composite reversal signals
            df['potential_bullish_reversal'] = (df['overextended_down'] | 
                                              df['bullish_divergence'] | 
                                              df['declining_down_volume'] | 
                                              df['momentum_weakening_down'] |
                                              df['hammer']).astype(int)
            
            df['potential_bearish_reversal'] = (df['overextended_up'] | 
                                              df['bearish_divergence'] | 
                                              df['declining_up_volume'] | 
                                              df['momentum_weakening_up'] |
                                              df['shooting_star']).astype(int)
            
            # 9. Add strength ranking to the signals (0-5 scale)
            df['bullish_reversal_strength'] = 0
            df['bearish_reversal_strength'] = 0
            
            # Count how many bullish signals are present
            for signal in ['overextended_down', 'bullish_divergence', 'declining_down_volume', 
                          'momentum_weakening_down', 'hammer']:
                if signal in df.columns:
                    df['bullish_reversal_strength'] += df[signal].astype(int)
            
            # Count how many bearish signals are present  
            for signal in ['overextended_up', 'bearish_divergence', 'declining_up_volume', 
                          'momentum_weakening_up', 'shooting_star']:
                if signal in df.columns:
                    df['bearish_reversal_strength'] += df[signal].astype(int)
                    
            logger.info("Added reversal detection indicators")
            
        except Exception as e:
            logger.error(f"Error adding reversal indicators: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    @staticmethod
    def check_for_reversals(df: pd.DataFrame, 
                           current_signal: str, 
                           min_reversal_strength: int = 2) -> Tuple[bool, str, float]:
        """
        Check if current market conditions suggest a reversal
        
        Args:
            df: DataFrame with technical indicators
            current_signal: Current trading signal (LONG/SHORT/NEUTRAL)
            min_reversal_strength: Minimum strength required to signal reversal (0-5)
            
        Returns:
            Tuple of (reversal_detected, new_signal, confidence)
        """
        if len(df) < 5:
            return False, current_signal, 0
        
        # Get most recent row
        last_row = df.iloc[-1]
        
        # Default values
        reversal_detected = False
        new_signal = current_signal
        confidence = 0
        
        try:
            # Check for reversal based on current signal
            if current_signal == "LONG":
                # Check for bearish reversal signals
                if ('bearish_reversal_strength' in last_row and 
                    last_row['bearish_reversal_strength'] >= min_reversal_strength):
                    # Calculate distance from recent high
                    recent_high = df['high'].tail(10).max()
                    distance_from_high = (recent_high - last_row['close']) / recent_high * 100
                    
                    # Only trigger if we've moved away from the high by some amount
                    if distance_from_high > 3:  # More than 3% from recent high
                        reversal_detected = True
                        new_signal = "NEUTRAL"  # or could be "SHORT" for more aggressive flipping
                        confidence = min(100, last_row['bearish_reversal_strength'] * 20)
                        
                        logger.info(f"Bearish reversal detected with strength {last_row['bearish_reversal_strength']}. "
                                  f"Current price is {distance_from_high:.2f}% below recent high.")
            
            elif current_signal == "SHORT":
                # Check for bullish reversal signals
                if ('bullish_reversal_strength' in last_row and 
                    last_row['bullish_reversal_strength'] >= min_reversal_strength):
                    # Calculate distance from recent low
                    recent_low = df['low'].tail(10).min()
                    distance_from_low = (last_row['close'] - recent_low) / recent_low * 100
                    
                    # Only trigger if we've moved away from the low by some amount
                    if distance_from_low > 3:  # More than 3% from recent low
                        reversal_detected = True
                        new_signal = "NEUTRAL"  # or could be "LONG" for more aggressive flipping
                        confidence = min(100, last_row['bullish_reversal_strength'] * 20)
                        
                        logger.info(f"Bullish reversal detected with strength {last_row['bullish_reversal_strength']}. "
                                  f"Current price is {distance_from_low:.2f}% above recent low.")
            
            # If currently neutral, check both directions for potential new signals
            elif current_signal == "NEUTRAL":
                # Check bullish conditions
                if ('bullish_reversal_strength' in last_row and 
                    last_row['bullish_reversal_strength'] >= min_reversal_strength + 1):  # Higher threshold for new signals
                    reversal_detected = True
                    new_signal = "LONG"
                    confidence = min(100, last_row['bullish_reversal_strength'] * 15)
                    
                # Check bearish conditions
                elif ('bearish_reversal_strength' in last_row and 
                      last_row['bearish_reversal_strength'] >= min_reversal_strength + 1):  # Higher threshold for new signals
                    reversal_detected = True
                    new_signal = "SHORT"
                    confidence = min(100, last_row['bearish_reversal_strength'] * 15)
        
        except Exception as e:
            logger.error(f"Error checking for reversals: {e}")
            import traceback
            traceback.print_exc()
        
        return reversal_detected, new_signal, confidence