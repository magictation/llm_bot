"""
Advanced technical analysis module using pandas-ta instead of TA-Lib
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
# Import pandas_ta for technical indicators
import pandas_ta as ta
from app.config import logger
from app.analysis.technical import TechnicalAnalyzer

class AdvancedTechnicalAnalyzer:
    """Enhanced technical analysis with advanced indicators and pattern recognition"""
    
    @staticmethod
    def add_standard_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add standard technical indicators to DataFrame (same as original)"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        try:
            # Calculate moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # Calculate RSI using pandas-ta
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # Calculate MACD using pandas-ta - handle return value properly
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if isinstance(macd_result, pd.DataFrame):
                # Get column names from the result
                macd_cols = macd_result.columns.tolist()
                if len(macd_cols) >= 3:
                    df['macd'] = macd_result[macd_cols[0]]
                    df['macd_signal'] = macd_result[macd_cols[1]]
                    df['macd_histogram'] = macd_result[macd_cols[2]]
                else:
                    # Fallback if columns are missing
                    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_histogram'] = df['macd'] - df['macd_signal']
            else:
                # Fallback if return is not a DataFrame
                df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Calculate Average True Range (ATR) using pandas-ta
            atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
            if isinstance(atr_result, pd.Series):
                df['atr'] = atr_result
            else:
                # Fallback calculation
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df['atr'] = true_range.rolling(14).mean()
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            
            # Price rate of change
            roc_result = ta.roc(df['close'], length=10)
            if isinstance(roc_result, pd.Series):
                df['roc'] = roc_result
            else:
                # Fallback calculation
                df['roc'] = df['close'].pct_change(periods=10) * 100
        
        except Exception as e:
            logger.error(f"Error in add_standard_indicators: {e}")
            # Use basic indicators
            df = TechnicalAnalyzer.add_indicators(df)
        
        return df
    
    @staticmethod
    def add_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators to DataFrame using pandas-ta"""
        df = df.copy()
        
        try:
            # Calculate manually instead of using pandas-ta which can return unexpected formats
            # Conversion Line (Tenkan-sen) - 9-period moving average of midpoint price
            period9_high = df['high'].rolling(window=9).max()
            period9_low = df['low'].rolling(window=9).min()
            df['tenkan_sen'] = (period9_high + period9_low) / 2
            
            # Base Line (Kijun-sen) - 26-period moving average of midpoint price
            period26_high = df['high'].rolling(window=26).max()
            period26_low = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (period26_high + period26_low) / 2
            
            # Leading Span A (Senkou Span A) - Average of conversion and base lines, shifted 26 periods forward
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            # Leading Span B (Senkou Span B) - 52-period moving average of midpoint price, shifted 26 periods forward
            period52_high = df['high'].rolling(window=52).max()
            period52_low = df['low'].rolling(window=52).min()
            df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
            
            # Lagging Span (Chikou Span) - Current close shifted 26 periods backward
            df['chikou_span'] = df['close'].shift(-26)
            
            # Cloud edges and color
            df['cloud_green'] = df['senkou_span_a'] > df['senkou_span_b']
            
            # Ichimoku signals
            df['price_above_cloud'] = (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])
            df['price_below_cloud'] = (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])
            df['tenkan_kijun_cross_bullish'] = (df['tenkan_sen'] > df['kijun_sen']) & (df['tenkan_sen'].shift() <= df['kijun_sen'].shift())
            df['tenkan_kijun_cross_bearish'] = (df['tenkan_sen'] < df['kijun_sen']) & (df['tenkan_sen'].shift() >= df['kijun_sen'].shift())
        
        except Exception as e:
            logger.warning(f"Error calculating Ichimoku Cloud: {e}")
            # Create empty columns
            ichimoku_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
                               'cloud_green', 'price_above_cloud', 'price_below_cloud', 
                               'tenkan_kijun_cross_bullish', 'tenkan_kijun_cross_bearish']
            for col in ichimoku_columns:
                if col not in df.columns:
                    df[col] = np.nan
            
        return df
    
    @staticmethod
    def add_fibonacci_retracements(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """
        Add Fibonacci retracement levels based on recent swing high/low
        
        Args:
            df: DataFrame with OHLCV data
            window: Window to look for swing high/low
        """
        df = df.copy()
        
        # Initialize fib level columns first
        fib_columns = ['swing_high', 'swing_low', 'fib_0', 'fib_0.236', 'fib_0.382', 
                      'fib_0.5', 'fib_0.618', 'fib_0.786', 'fib_1']
        
        for col in fib_columns:
            df[col] = np.nan
            
        # Initialize signal columns
        df['at_fib_0.618'] = False
        df['at_fib_0.5'] = False
        df['at_fib_0.382'] = False
        
        try:
            # Find local maxima and minima
            df['is_local_max'] = df['high'].rolling(window=window, center=True).apply(
                lambda x: x.argmax() == len(x) // 2, raw=False)
            df['is_local_min'] = df['low'].rolling(window=window, center=True).apply(
                lambda x: x.argmin() == len(x) // 2, raw=False)
                
            # Function to calculate Fibonacci levels for each row
            def calculate_fib_levels(row_idx):
                # Default values
                fib_levels = {
                    'swing_high': np.nan, 
                    'swing_low': np.nan,
                    'fib_0': np.nan, 
                    'fib_0.236': np.nan, 
                    'fib_0.382': np.nan,
                    'fib_0.5': np.nan, 
                    'fib_0.618': np.nan, 
                    'fib_0.786': np.nan, 
                    'fib_1': np.nan
                }
                
                if row_idx < window:
                    return fib_levels
                
                # Look back for swing high and low
                prev_data = df.iloc[max(0, row_idx-window):row_idx]
                
                # Find last swing high and low
                swing_highs = prev_data[prev_data['is_local_max'] == True]
                swing_lows = prev_data[prev_data['is_local_min'] == True]
                
                if len(swing_highs) == 0 or len(swing_lows) == 0:
                    return fib_levels
                
                last_swing_high = swing_highs.iloc[-1]['high']
                last_swing_low = swing_lows.iloc[-1]['low']
                
                # Determine if we're in an uptrend or downtrend based on the most recent swing
                if swing_highs.index[-1] > swing_lows.index[-1]:
                    # Uptrend, calculate retracements from low to high
                    fib_range = last_swing_high - last_swing_low
                    fib_levels = {
                        'swing_high': last_swing_high,
                        'swing_low': last_swing_low,
                        'fib_0': last_swing_low,
                        'fib_0.236': last_swing_low + 0.236 * fib_range,
                        'fib_0.382': last_swing_low + 0.382 * fib_range,
                        'fib_0.5': last_swing_low + 0.5 * fib_range,
                        'fib_0.618': last_swing_low + 0.618 * fib_range,
                        'fib_0.786': last_swing_low + 0.786 * fib_range,
                        'fib_1': last_swing_high
                    }
                else:
                    # Downtrend, calculate retracements from high to low
                    fib_range = last_swing_high - last_swing_low
                    fib_levels = {
                        'swing_high': last_swing_high,
                        'swing_low': last_swing_low,
                        'fib_0': last_swing_high,
                        'fib_0.236': last_swing_high - 0.236 * fib_range,
                        'fib_0.382': last_swing_high - 0.382 * fib_range,
                        'fib_0.5': last_swing_high - 0.5 * fib_range,
                        'fib_0.618': last_swing_high - 0.618 * fib_range,
                        'fib_0.786': last_swing_high - 0.786 * fib_range,
                        'fib_1': last_swing_low
                    }
                    
                return fib_levels
            
            # Update values for each row
            for i in range(len(df)):
                fib_levels = calculate_fib_levels(i)
                for col in fib_columns:
                    df.loc[df.index[i], col] = fib_levels[col]
            
            # Fib level signals
            df['at_fib_0.618'] = (df['low'] <= df['fib_0.618']) & (df['high'] >= df['fib_0.618'])
            df['at_fib_0.5'] = (df['low'] <= df['fib_0.5']) & (df['high'] >= df['fib_0.5'])
            df['at_fib_0.382'] = (df['low'] <= df['fib_0.382']) & (df['high'] >= df['fib_0.382'])
        except Exception as e:
            logger.warning(f"Error calculating Fibonacci levels: {e}")
                
        return df
    
    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect common chart patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern detection columns
        """
        df = df.copy()
        
        # Initialize pattern columns
        pattern_columns = [
            'double_top', 'double_bottom', 'head_shoulders', 'inv_head_shoulders',
            'triangle_ascending', 'triangle_descending', 'triangle_symmetrical',
            'flag_bullish', 'flag_bearish'
        ]
        
        for col in pattern_columns:
            df[col] = 0
        
        try:
            # Simplified pattern detection
            # Pre-defined window sizes for pattern detection
            window_small = 5
            window_medium = 14
            window_large = 30
            
            # Detect Double Top
            for i in range(window_large, len(df)):
                section = df.iloc[i-window_large:i]
                # Find local maxima
                max_indices = argrelextrema(section['high'].values, np.greater, order=window_small)[0]
                
                if len(max_indices) >= 2:
                    # Get the highest two peaks
                    peak_values = [section.iloc[idx]['high'] for idx in max_indices]
                    sorted_peaks = sorted(zip(max_indices, peak_values), key=lambda x: x[1], reverse=True)
                    
                    if len(sorted_peaks) >= 2:
                        # Check if the top two peaks are close in height
                        peak1, val1 = sorted_peaks[0]
                        peak2, val2 = sorted_peaks[1]
                        
                        height_diff_pct = abs(val1 - val2) / val1
                        if height_diff_pct < 0.03 and abs(peak1 - peak2) > window_small:
                            df.iloc[i, df.columns.get_loc('double_top')] = 1
            
            # Detect Double Bottom (similar to Double Top but for lows)
            for i in range(window_large, len(df)):
                section = df.iloc[i-window_large:i]
                # Find local minima
                min_indices = argrelextrema(section['low'].values, np.less, order=window_small)[0]
                
                if len(min_indices) >= 2:
                    # Get the lowest two troughs
                    trough_values = [section.iloc[idx]['low'] for idx in min_indices]
                    sorted_troughs = sorted(zip(min_indices, trough_values), key=lambda x: x[1])
                    
                    if len(sorted_troughs) >= 2:
                        # Check if the bottom two troughs are close in height
                        trough1, val1 = sorted_troughs[0]
                        trough2, val2 = sorted_troughs[1]
                        
                        height_diff_pct = abs(val1 - val2) / val1
                        if height_diff_pct < 0.03 and abs(trough1 - trough2) > window_small:
                            df.iloc[i, df.columns.get_loc('double_bottom')] = 1
        except Exception as e:
            logger.warning(f"Error during pattern detection: {e}")
        
        return df
    
    @staticmethod
    def add_volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
        """
        Add volume profile analysis
        
        Args:
            df: DataFrame with OHLCV data
            bins: Number of price bins for volume profile
            
        Returns:
            DataFrame with volume profile columns
        """
        df = df.copy()
        
        # Initialize volume profile columns
        df['vp_poc'] = df['close']
        df['vp_value_area_low'] = df['low']
        df['vp_value_area_high'] = df['high']
        df['price_at_poc'] = False
        df['price_below_value_area'] = False
        df['price_above_value_area'] = False
        df['price_in_value_area'] = True
        
        try:
            # Calculate the full range for the volume profile
            min_price = df['low'].min()
            max_price = df['high'].max()
            
            if min_price == max_price:
                logger.warning("Price range is zero, skipping volume profile")
                return df
            
            # Create price bins
            price_range = max_price - min_price
            bin_size = price_range / bins
            
            # Function to calculate which bins a candle occupies and assign volume proportionally
            def candle_volume_distribution(row):
                # Calculate how many bins this candle spans
                candle_min_bin = max(0, int((row['low'] - min_price) / bin_size))
                candle_max_bin = min(bins-1, int((row['high'] - min_price) / bin_size))
                
                # Initialize distribution with zeros
                distribution = [0] * bins
                
                # If candle spans multiple bins, distribute volume proportionally
                if candle_max_bin > candle_min_bin:
                    span = candle_max_bin - candle_min_bin + 1
                    vol_per_bin = row['volume'] / span
                    for i in range(candle_min_bin, candle_max_bin + 1):
                        distribution[i] = vol_per_bin
                else:
                    # If candle fits in one bin, assign all volume to that bin
                    distribution[candle_min_bin] = row['volume']
                    
                return distribution
            
            # Calculate volume distribution for all candles
            volume_distributions = df.apply(candle_volume_distribution, axis=1)
            
            # Sum volumes in each bin
            volume_profile = np.zeros(bins)
            for dist in volume_distributions:
                volume_profile += np.array(dist)
            
            # Find the Point of Control (price level with highest volume)
            poc_bin = np.argmax(volume_profile)
            poc_price = min_price + (poc_bin + 0.5) * bin_size
            
            # Find Value Area (70% of volume)
            total_volume = np.sum(volume_profile)
            target_volume = total_volume * 0.7
            
            # Start from POC and expand outward
            cumulative_volume = volume_profile[poc_bin]
            lower_bin = poc_bin
            upper_bin = poc_bin
            
            while cumulative_volume < target_volume and (lower_bin > 0 or upper_bin < bins - 1):
                # Expand to the bin with higher volume
                lower_vol = volume_profile[lower_bin - 1] if lower_bin > 0 else 0
                upper_vol = volume_profile[upper_bin + 1] if upper_bin < bins - 1 else 0
                
                if lower_vol > upper_vol and lower_bin > 0:
                    lower_bin -= 1
                    cumulative_volume += lower_vol
                elif upper_bin < bins - 1:
                    upper_bin += 1
                    cumulative_volume += upper_vol
                else:
                    break
            
            # Calculate Value Area boundaries
            value_area_low = min_price + lower_bin * bin_size
            value_area_high = min_price + (upper_bin + 1) * bin_size
            
            # Add to DataFrame
            df['vp_poc'] = poc_price
            df['vp_value_area_low'] = value_area_low
            df['vp_value_area_high'] = value_area_high
            
            # Add signals based on price relative to volume profile
            df['price_at_poc'] = (df['low'] <= poc_price) & (df['high'] >= poc_price)
            df['price_below_value_area'] = df['high'] < value_area_low
            df['price_above_value_area'] = df['low'] > value_area_high
            df['price_in_value_area'] = (df['low'] <= value_area_high) & (df['high'] >= value_area_low)
        
        except Exception as e:
            logger.warning(f"Error calculating volume profile: {e}")
        
        return df
    
    @staticmethod
    def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicators"""
        df = df.copy()
        
        try:
            # Hull Moving Average (simplified calculation)
            period = 20
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            
            # Calculate WMA(2*WMA(n/2) - WMA(n))
            half_wma = df['close'].rolling(window=half_period).mean()
            full_wma = df['close'].rolling(window=period).mean()
            raw_hma = 2 * half_wma - full_wma
            df['hma'] = raw_hma.rolling(window=sqrt_period).mean()
            
            # Directional Movement Index (DMI)
            # True Range
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Directional Movement
            df['up_move'] = df['high'] - df['high'].shift(1)
            df['down_move'] = df['low'].shift(1) - df['low']
            
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            
            # 14-period smoothed True Range and Directional Movement
            df['atr_14'] = df['true_range'].rolling(window=14).mean()
            df['plus_di_14'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr_14'])
            df['minus_di_14'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr_14'])
            
            # ADX calculation
            df['dx'] = 100 * abs(df['plus_di_14'] - df['minus_di_14']) / (df['plus_di_14'] + df['minus_di_14'])
            df['dx'] = df['dx'].replace([np.inf, -np.inf], np.nan).fillna(0)
            df['adx'] = df['dx'].rolling(window=14).mean()
            
            # Store with consistent names
            df['plus_di'] = df['plus_di_14']
            df['minus_di'] = df['minus_di_14']
            
            # Clean up temporary columns
            df = df.drop(['tr1', 'tr2', 'tr3', 'up_move', 'down_move', 'plus_dm', 
                         'minus_dm', 'atr_14', 'plus_di_14', 'minus_di_14', 'dx'], 
                         axis=1, errors='ignore')
            
        except Exception as e:
            logger.warning(f"Error adding custom indicators: {e}")
            
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to DataFrame with error handling"""
        try:
            df = AdvancedTechnicalAnalyzer.add_standard_indicators(df)
            df = AdvancedTechnicalAnalyzer.add_ichimoku_cloud(df)
            df = AdvancedTechnicalAnalyzer.add_fibonacci_retracements(df)
            df = AdvancedTechnicalAnalyzer.detect_chart_patterns(df)
            df = AdvancedTechnicalAnalyzer.add_volume_profile(df)
            df = AdvancedTechnicalAnalyzer.add_custom_indicators(df)
            
            # Make sure all columns exist by adding defaults for any missing ones
            required_columns = [
                'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
                'fib_0.382', 'fib_0.5', 'fib_0.618',
                'vp_poc', 'vp_value_area_low', 'vp_value_area_high',
                'double_top', 'double_bottom', 'head_shoulders', 'inv_head_shoulders'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = np.nan
            
            return df
        except Exception as e:
            logger.error(f"Error adding advanced indicators, falling back to basic: {e}")
            # Fall back to basic indicators if advanced ones fail
            return TechnicalAnalyzer.add_indicators(df)
    
    @staticmethod
    def detect_advanced_breakouts(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential breakouts based on advanced indicators with error handling
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with breakout signals
        """
        try:
            df = df.copy()
            
            # Calculate local highs and lows if not already present
            if 'local_high' not in df.columns:
                df['local_high'] = df['high'].rolling(window=20, center=False).max()
            if 'local_low' not in df.columns:
                df['local_low'] = df['low'].rolling(window=20, center=False).min()
            
            # Original breakout signals
            df['breaks_high'] = df['close'] > df['local_high'].shift(1)
            df['breaks_low'] = df['close'] < df['local_low'].shift(1)
            df['volume_surge'] = df['volume'] > df['volume_sma_20'] * 1.5
            df['vol_expansion'] = df['atr'] > df['atr'].shift(1) * 1.2
            
            # Ensure all required pattern columns exist
            pattern_cols = ['double_top', 'double_bottom', 'head_shoulders', 'inv_head_shoulders']
            for col in pattern_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # Ichimoku-based signals (check if Ichimoku columns exist)
            if all(col in df.columns for col in ['senkou_span_a', 'senkou_span_b', 'tenkan_sen', 'kijun_sen']):
                df['ichimoku_bullish'] = (
                    (df['close'] > df['senkou_span_a']) & 
                    (df['close'] > df['senkou_span_b']) & 
                    (df['tenkan_sen'] > df['kijun_sen'])
                )
                
                df['ichimoku_bearish'] = (
                    (df['close'] < df['senkou_span_a']) & 
                    (df['close'] < df['senkou_span_b']) & 
                    (df['tenkan_sen'] < df['kijun_sen'])
                )
            else:
                df['ichimoku_bullish'] = False
                df['ichimoku_bearish'] = False
            
            # Volume profile signals (check if volume profile columns exist)
            if all(col in df.columns for col in ['vp_value_area_high', 'vp_value_area_low']):
                df['vp_breakout_up'] = (
                    (df['close'] > df['vp_value_area_high']) & 
                    (df['close'].shift(1) <= df['vp_value_area_high']) & 
                    (df['volume'] > df['volume_sma_20'])
                )
                
                df['vp_breakout_down'] = (
                    (df['close'] < df['vp_value_area_low']) & 
                    (df['close'].shift(1) >= df['vp_value_area_low']) & 
                    (df['volume'] > df['volume_sma_20'])
                )
            else:
                df['vp_breakout_up'] = False
                df['vp_breakout_down'] = False
            
            # Pattern-based signals
            df['pattern_bullish'] = ((df['double_bottom'] == 1) | (df['inv_head_shoulders'] == 1))
            df['pattern_bearish'] = ((df['double_top'] == 1) | (df['head_shoulders'] == 1))
            
            # Fibonacci retracement signals (check if Fibonacci columns exist)
            if all(col in df.columns for col in ['fib_0.618', 'fib_0.382']):
                df['fib_bounce_up'] = (
                    (df['low'] <= df['fib_0.618']) & 
                    (df['close'] > df['open']) & 
                    (df['close'] > df['fib_0.618'])
                )
                
                df['fib_bounce_down'] = (
                    (df['high'] >= df['fib_0.382']) & 
                    (df['close'] < df['open']) & 
                    (df['close'] < df['fib_0.382'])
                )
            else:
                df['fib_bounce_up'] = False
                df['fib_bounce_down'] = False
            
            # Combined advanced signals
            df['advanced_long_signal'] = (
                # Original breakout conditions
                (df['breaks_high'] & df['volume_surge'] & df['vol_expansion']) |
                # Ichimoku cloud breakout
                (df['ichimoku_bullish'] & df['volume_surge']) |
                # Volume profile breakout
                (df['vp_breakout_up']) |
                # Pattern and Fibonacci combination
                (df['pattern_bullish'] & df['fib_bounce_up'] & df['volume_surge'])
            ).astype(int)
            
            df['advanced_short_signal'] = (
                # Original breakout conditions
                (df['breaks_low'] & df['volume_surge'] & df['vol_expansion']) |
                # Ichimoku cloud breakdown
                (df['ichimoku_bearish'] & df['volume_surge']) |
                # Volume profile breakdown
                (df['vp_breakout_down']) |
                # Pattern and Fibonacci combination
                (df['pattern_bearish'] & df['fib_bounce_down'] & df['volume_surge'])
            ).astype(int)
            
            return df
        
        except Exception as e:
            logger.error(f"Error detecting advanced breakouts, falling back to basic: {e}")
            # Fall back to basic breakout detection if advanced ones fail
            return TechnicalAnalyzer.detect_breakouts(df)
    
    @staticmethod
    def prepare_advanced_data_for_llm(df: pd.DataFrame, rows: int = 50) -> str:
        """
        Prepare a subset of advanced indicator data for LLM analysis
        
        Args:
            df: DataFrame with indicators
            rows: Number of recent rows to include
            
        Returns:
            CSV string of the subset data
        """
        try:
            # Select relevant columns for analysis
            relevant_columns = [
                'open', 'high', 'low', 'close', 'volume', 
                'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 
                'bb_upper', 'bb_lower', 'atr',
                # Ichimoku columns
                'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
                # Fibonacci levels
                'fib_0.382', 'fib_0.5', 'fib_0.618',
                # Volume profile
                'vp_poc', 'vp_value_area_low', 'vp_value_area_high',
                # Custom indicators
                'adx', 'plus_di', 'minus_di', 'hma',
                # Signals
                'advanced_long_signal', 'advanced_short_signal'
            ]
            
            # Filter out columns that don't exist
            existing_columns = [col for col in relevant_columns if col in df.columns]
            
            # Get the most recent rows
            subset = df.tail(rows)[existing_columns]
            
            # Convert to CSV string
            csv_string = subset.to_csv()
            return csv_string
        except Exception as e:
            logger.error(f"Error preparing advanced data for LLM: {e}")
            # Fall back to basic data preparation if advanced fails
            return TechnicalAnalyzer.prepare_data_for_llm(df, rows)