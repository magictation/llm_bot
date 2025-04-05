"""
Multi-timeframe analysis module for Bitcoin Trading Bot
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from app.analysis.technical import TechnicalAnalyzer
from app.config import logger

class MultiTimeframeAnalyzer:
    """Class to perform multi-timeframe analysis on price data"""
    
    def __init__(self, data_fetcher, timeframes: List[str] = None):
        """
        Initialize with data fetcher and timeframes
        
        Args:
            data_fetcher: Instance of BinanceDataFetcher
            timeframes: List of timeframes to analyze (e.g., ['5m', '15m', '1h', '4h'])
        """
        self.data_fetcher = data_fetcher
        self.timeframes = timeframes or ['5m', '15m', '1h', '4h']
        logger.info(f"Initialized Multi-Timeframe Analyzer with timeframes: {self.timeframes}")
    
    def fetch_all_timeframes(self, symbol: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all timeframes
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary of timeframes to DataFrames
        """
        import datetime
        
        # Calculate start date
        start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # Fetch data for each timeframe
        timeframe_data = {}
        
        for timeframe in self.timeframes:
            try:
                logger.info(f"Fetching {timeframe} data for {symbol}...")
                df = self.data_fetcher.get_historical_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=start_date
                )
                
                # Add technical indicators
                df = TechnicalAnalyzer.add_indicators(df)
                
                # Detect breakouts
                df = TechnicalAnalyzer.detect_breakouts(df)
                
                timeframe_data[timeframe] = df
                logger.info(f"Successfully fetched and analyzed {len(df)} rows of {timeframe} data")
                
            except Exception as e:
                logger.error(f"Error fetching {timeframe} data: {e}")
                # Continue with other timeframes if one fails
                continue
                
        return timeframe_data
    
    def analyze_timeframe_data(self, data: pd.DataFrame, importance: float = 1.0) -> Dict:
        """
        Analyze a single timeframe for signals
        
        Args:
            data: DataFrame with indicators
            importance: Weight of this timeframe (higher for longer timeframes)
            
        Returns:
            Dictionary with signal analysis
        """
        # Make sure we have data to analyze
        if data is None or len(data) == 0:
            return {
                "bullish_signals": [],
                "bullish_score": 0,
                "weighted_bullish": 0,
                "bearish_signals": [],
                "bearish_score": 0,
                "weighted_bearish": 0,
                "neutral_signals": [],
                "signal": "NEUTRAL",
                "confidence": 0
            }
            
        # Get the most recent data
        last_row = data.iloc[-1]
        
        # Calculate bullish signals
        bullish_signals = []
        bullish_score = 0
        
        # Moving averages
        if 'sma_20' in last_row and 'sma_50' in last_row:
            if last_row['close'] > last_row['sma_20']:
                bullish_signals.append("Price above SMA-20")
                bullish_score += 1
                
            if last_row['close'] > last_row['sma_50']:
                bullish_signals.append("Price above SMA-50")
                bullish_score += 1
                
            if last_row['sma_20'] > last_row['sma_50']:
                bullish_signals.append("SMA-20 above SMA-50 (Golden Cross potential)")
                bullish_score += 2
        
        # RSI
        if 'rsi' in last_row:
            if 30 < last_row['rsi'] < 70:
                if last_row['rsi'] > 50:
                    bullish_signals.append("RSI above 50 (bullish momentum)")
                    bullish_score += 1
            elif last_row['rsi'] <= 30:
                bullish_signals.append("RSI oversold (potential bounce)")
                bullish_score += 1
            
        # MACD
        if 'macd' in last_row and 'macd_signal' in last_row:
            if last_row['macd'] > last_row['macd_signal']:
                bullish_signals.append("MACD above Signal Line")
                bullish_score += 2
                
            # Check if previous row exists for this comparison
            if len(data) > 1 and 'macd' in data.iloc[-2]:
                prev_macd = data.iloc[-2]['macd']
                if last_row['macd'] < 0 and last_row['macd'] > prev_macd:
                    bullish_signals.append("MACD rising while below zero")
                    bullish_score += 1
            
        # Bollinger Bands
        if 'bb_lower' in last_row:
            if last_row['close'] < last_row['bb_lower']:
                bullish_signals.append("Price below lower Bollinger Band (oversold)")
                bullish_score += 1
            
        # Breakout signal
        if 'long_signal' in last_row:
            if last_row['long_signal'] == 1:
                bullish_signals.append("Long breakout signal triggered")
                bullish_score += 3
            
        # Calculate bearish signals
        bearish_signals = []
        bearish_score = 0
        
        # Moving averages
        if 'sma_20' in last_row and 'sma_50' in last_row:
            if last_row['close'] < last_row['sma_20']:
                bearish_signals.append("Price below SMA-20")
                bearish_score += 1
                
            if last_row['close'] < last_row['sma_50']:
                bearish_signals.append("Price below SMA-50")
                bearish_score += 1
                
            if last_row['sma_20'] < last_row['sma_50']:
                bearish_signals.append("SMA-20 below SMA-50 (Death Cross potential)")
                bearish_score += 2
            
        # RSI
        if 'rsi' in last_row:
            if 30 < last_row['rsi'] < 70:
                if last_row['rsi'] < 50:
                    bearish_signals.append("RSI below 50 (bearish momentum)")
                    bearish_score += 1
            elif last_row['rsi'] >= 70:
                bearish_signals.append("RSI overbought (potential reversal)")
                bearish_score += 1
            
        # MACD
        if 'macd' in last_row and 'macd_signal' in last_row:
            if last_row['macd'] < last_row['macd_signal']:
                bearish_signals.append("MACD below Signal Line")
                bearish_score += 2
            
            # Check if previous row exists for this comparison
            if len(data) > 1 and 'macd' in data.iloc[-2]:
                prev_macd = data.iloc[-2]['macd']
                if last_row['macd'] > 0 and last_row['macd'] < prev_macd:
                    bearish_signals.append("MACD falling while above zero")
                    bearish_score += 1
            
        # Bollinger Bands
        if 'bb_upper' in last_row:
            if last_row['close'] > last_row['bb_upper']:
                bearish_signals.append("Price above upper Bollinger Band (overbought)")
                bearish_score += 1
            
        # Breakout signal
        if 'short_signal' in last_row:
            if last_row['short_signal'] == 1:
                bearish_signals.append("Short breakout signal triggered")
                bearish_score += 3
            
        # Calculate neutral signals
        neutral_signals = []
        
        # Price within Bollinger Bands
        if 'bb_lower' in last_row and 'bb_upper' in last_row:
            if (last_row['close'] > last_row['bb_lower']) and (last_row['close'] < last_row['bb_upper']):
                neutral_signals.append("Price within Bollinger Bands")
            
        # RSI in middle range
        if 'rsi' in last_row:
            if 40 < last_row['rsi'] < 60:
                neutral_signals.append("RSI in neutral zone")
            
        # MACD near zero
        if 'macd' in last_row:
            if abs(last_row['macd']) < 0.1 * last_row['close']:
                neutral_signals.append("MACD near zero line")
            
        # Determine overall signal
        weighted_bullish = bullish_score * importance
        weighted_bearish = bearish_score * importance
        
        if weighted_bullish > weighted_bearish * 1.5:
            signal = "LONG"
            confidence = min(100, (weighted_bullish / (weighted_bullish + weighted_bearish)) * 100) if (weighted_bullish + weighted_bearish) > 0 else 50
        elif weighted_bearish > weighted_bullish * 1.5:
            signal = "SHORT"
            confidence = min(100, (weighted_bearish / (weighted_bullish + weighted_bearish)) * 100) if (weighted_bullish + weighted_bearish) > 0 else 50
        else:
            signal = "NEUTRAL"
            difference = abs(weighted_bullish - weighted_bearish)
            max_score = max(weighted_bullish, weighted_bearish)
            confidence = min(100, 50 - (difference / max_score) * 50) if max_score > 0 else 50
        
        return {
            "bullish_signals": bullish_signals,
            "bullish_score": bullish_score,
            "weighted_bullish": weighted_bullish,
            "bearish_signals": bearish_signals,
            "bearish_score": bearish_score,
            "weighted_bearish": weighted_bearish,
            "neutral_signals": neutral_signals,
            "signal": signal,
            "confidence": confidence
        }
    
    def combine_timeframe_signals(self, timeframe_analyses: Dict[str, Dict]) -> Dict:
        """
        Combine signals from multiple timeframes into a single analysis
        
        Args:
            timeframe_analyses: Dictionary of timeframe to analysis results
            
        Returns:
            Dictionary with combined analysis
        """
        if not timeframe_analyses:
            logger.warning("No timeframe analyses to combine")
            return {
                "signal": "NEUTRAL",
                "confidence": 0,
                "timeframes_agreeing": 0,
                "total_timeframes": 0,
                "primary_bullish_signals": [],
                "primary_bearish_signals": [],
                "all_timeframe_signals": {}
            }
        
        # Count signals by type
        signal_counts = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        
        # Weight factors for different timeframes (longer timeframes get higher weight)
        timeframe_weights = {
            '1m': 0.5,
            '3m': 0.6,
            '5m': 0.7,
            '15m': 0.8,
            '30m': 0.9,
            '1h': 1.0,
            '2h': 1.1,
            '4h': 1.2,
            '6h': 1.3,
            '8h': 1.4,
            '12h': 1.5,
            '1d': 1.7,
            '3d': 1.8,
            '1w': 2.0,
            '1M': 2.5
        }
        
        # Default weight if timeframe not in the map
        default_weight = 1.0
        
        # Collect weighted votes
        weighted_votes = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        confidence_sum = 0
        weight_sum = 0
        
        # Collect all signals
        all_bullish_signals = []
        all_bearish_signals = []
        all_timeframe_signals = {}
        
        for timeframe, analysis in timeframe_analyses.items():
            signal = analysis.get("signal", "NEUTRAL")
            confidence = analysis.get("confidence", 0)
            
            # Get weight for this timeframe
            weight = timeframe_weights.get(timeframe, default_weight)
            
            # Update counts
            signal_counts[signal] += 1
            
            # Update weighted votes
            weighted_votes[signal] += weight * (confidence / 100)
            
            # Update confidence sum and weight sum for weighted average
            confidence_sum += confidence * weight
            weight_sum += weight
            
            # Collect signals
            all_bullish_signals.extend(analysis.get("bullish_signals", []))
            all_bearish_signals.extend(analysis.get("bearish_signals", []))
            
            # Store in all timeframe signals
            all_timeframe_signals[timeframe] = {
                "signal": signal,
                "confidence": confidence,
                "bullish": analysis.get("bullish_signals", []),
                "bearish": analysis.get("bearish_signals", []),
                "neutral": analysis.get("neutral_signals", [])
            }
        
        # Determine overall signal based on weighted votes
        if weighted_votes["LONG"] > weighted_votes["SHORT"] and weighted_votes["LONG"] > weighted_votes["NEUTRAL"]:
            overall_signal = "LONG"
        elif weighted_votes["SHORT"] > weighted_votes["LONG"] and weighted_votes["SHORT"] > weighted_votes["NEUTRAL"]:
            overall_signal = "SHORT"
        else:
            overall_signal = "NEUTRAL"
        
        # Calculate weighted average confidence
        if weight_sum > 0:
            overall_confidence = confidence_sum / weight_sum
        else:
            overall_confidence = 0
        
        # Calculate agreement score - what percentage of timeframes agree with the overall signal
        agreeing_timeframes = signal_counts[overall_signal]
        total_timeframes = len(timeframe_analyses)
        agreement_score = (agreeing_timeframes / total_timeframes) * 100 if total_timeframes > 0 else 0
        
        # Count occurrences of each signal
        from collections import Counter
        
        bullish_signal_counts = Counter(all_bullish_signals)
        bearish_signal_counts = Counter(all_bearish_signals)
        
        # Get top signals
        top_bullish = [signal for signal, count in bullish_signal_counts.most_common(5)]
        top_bearish = [signal for signal, count in bearish_signal_counts.most_common(5)]
        
        return {
            "signal": overall_signal,
            "confidence": overall_confidence,
            "agreement_score": agreement_score,
            "timeframes_agreeing": agreeing_timeframes,
            "total_timeframes": total_timeframes,
            "weighted_votes": weighted_votes,
            "primary_bullish_signals": top_bullish,
            "primary_bearish_signals": top_bearish,
            "all_timeframe_signals": all_timeframe_signals
        }
    
    def get_multi_timeframe_analysis(self, symbol: str, lookback_days: int) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Get combined analysis from multiple timeframes
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            lookback_days: Number of days to look back
            
        Returns:
            Tuple of (timeframe_data, combined_analysis)
        """
        # Fetch data for all timeframes
        timeframe_data = self.fetch_all_timeframes(symbol, lookback_days)
        
        if not timeframe_data:
            logger.error("Failed to fetch data for any timeframe")
            return {}, {
                "signal": "NEUTRAL",
                "confidence": 0,
                "error": "Failed to fetch timeframe data"
            }
        
        # Analyze each timeframe
        timeframe_analyses = {}
        
        for timeframe, df in timeframe_data.items():
            # Calculate importance based on timeframe
            # Longer timeframes get higher importance
            importance = 1.0
            if timeframe == '1h':
                importance = 1.2
            elif timeframe == '4h':
                importance = 1.5
            elif timeframe == '1d':
                importance = 2.0
            elif timeframe == '1w':
                importance = 2.5
            
            # Perform analysis
            analysis = self.analyze_timeframe_data(df, importance)
            timeframe_analyses[timeframe] = analysis
        
        # Combine analyses
        combined_analysis = self.combine_timeframe_signals(timeframe_analyses)
        
        return timeframe_data, combined_analysis
    
    def prepare_multi_timeframe_summary_for_llm(self, combined_analysis: Dict) -> str:
        """
        Prepare a summary of multi-timeframe analysis for LLM
        
        Args:
            combined_analysis: Combined analysis from multiple timeframes
            
        Returns:
            Formatted string summary
        """
        summary = f"""MULTI-TIMEFRAME ANALYSIS SUMMARY:

Overall Signal: {combined_analysis.get('signal', 'NEUTRAL')}
Confidence: {combined_analysis.get('confidence', 0):.2f}%
Agreement: {combined_analysis.get('agreement_score', 0):.2f}% ({combined_analysis.get('timeframes_agreeing', 0)} out of {combined_analysis.get('total_timeframes', 0)} timeframes)

Primary Bullish Signals:
{chr(10).join(['- ' + signal for signal in combined_analysis.get('primary_bullish_signals', [])])}

Primary Bearish Signals:
{chr(10).join(['- ' + signal for signal in combined_analysis.get('primary_bearish_signals', [])])}

Timeframe Breakdown:
"""
        
        for timeframe, analysis in combined_analysis.get('all_timeframe_signals', {}).items():
            summary += f"\n{timeframe}: {analysis.get('signal', 'NEUTRAL')} ({analysis.get('confidence', 0):.2f}% confidence)"
        
        return summary