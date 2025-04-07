"""
Enhanced LLM-based analysis module with improved prompting and reasoning
"""
import os
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import io
import google.generativeai as genai
from app.config import logger

class EnhancedLLMAnalyzer:
    """Enhanced class to interact with LLM for trading insights with multi-step reasoning"""
    
    def __init__(self, llm_api_key: str):
        """
        Initialize LLM analyzer with API key
        
        Args:
            llm_api_key: API key for the LLM service
        """
        # Prioritize the passed API key over environment variables
        api_key = llm_api_key or os.getenv('GEMINI_API_KEY') or os.getenv('LLM_API_KEY')
        
        if not api_key:
            raise ValueError("No API key provided for LLM")
        
        # Clear any existing environment variable to ensure fresh configuration
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        
        # Explicitly set the API key
        os.environ["GEMINI_API_KEY"] = api_key
        
        # Initialize Google AI client
        genai.configure(api_key=api_key)
        # Use the gemma-3-27b-it model
        self.model_name = "gemini-2.0-flash"
        logger.info(f"Initialized Enhanced LLM Analyzer with model: {self.model_name}")
    
    def prepare_comprehensive_data(self, df: pd.DataFrame) -> Dict:
        """Prepare comprehensive data summary for the LLM"""
        # Get the most recent data points
        latest = df.iloc[-1]
        
        # Price action summary
        price_change_24h = ((latest['close'] / df.iloc[-96]['close']) - 1) * 100 if len(df) > 96 else 0
        price_change_1h = ((latest['close'] / df.iloc[-4]['close']) - 1) * 100 if len(df) > 4 else 0
        price_change_5m = ((latest['close'] / df.iloc[-1]['close']) - 1) * 100 if len(df) > 1 else 0
        
        # Calculate volatility
        price_volatility = df['close'].pct_change().std() * 100  # Standard deviation of returns
        
        # Advanced price patterns (if available)
        price_patterns = {}
        for pattern in ['double_top', 'double_bottom', 'head_shoulders', 'inv_head_shoulders', 
                        'triangle_ascending', 'triangle_descending', 'hammer', 'shooting_star', 'doji']:
            if pattern in latest:
                price_patterns[pattern] = bool(latest[pattern])
        
        # Check for reversal indicators
        reversal_indicators = {}
        for indicator in ['potential_bullish_reversal', 'potential_bearish_reversal', 
                         'bullish_reversal_strength', 'bearish_reversal_strength',
                         'bullish_divergence', 'bearish_divergence']:
            if indicator in latest:
                reversal_indicators[indicator] = latest[indicator]
        
        # Volume analysis
        volume_metrics = {
            'current_volume': latest['volume'],
            'volume_sma_ratio': latest['volume'] / latest['volume_sma_20'] if 'volume_sma_20' in latest else 0,
        }
        
        if 'declining_up_volume' in latest:
            volume_metrics['declining_up_volume'] = bool(latest['declining_up_volume'])
        if 'declining_down_volume' in latest:
            volume_metrics['declining_down_volume'] = bool(latest['declining_down_volume'])
        
        # Technical indicator insights
        indicator_values = {
            'rsi': latest.get('rsi', 0),
            'macd': latest.get('macd', 0),
            'macd_signal': latest.get('macd_signal', 0),
            'macd_histogram': latest.get('macd_histogram', 0),
            'sma_20': latest.get('sma_20', 0),
            'sma_50': latest.get('sma_50', 0),
            'sma_200': latest.get('sma_200', 0),
            'ema_12': latest.get('ema_12', 0),
            'ema_26': latest.get('ema_26', 0),
            'bb_upper': latest.get('bb_upper', 0),
            'bb_middle': latest.get('bb_middle', 0),
            'bb_lower': latest.get('bb_lower', 0),
            'atr': latest.get('atr', 0),
        }
        
        # Advanced indicators (if available)
        advanced_indicators = {}
        for indicator in ['adx', 'plus_di', 'minus_di', 'hma', 'roc']:
            if indicator in latest:
                advanced_indicators[indicator] = latest[indicator]
        
        # Ichimoku data (if available)
        ichimoku_data = {}
        for element in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
                       'chikou_span', 'cloud_green', 'price_above_cloud', 'price_below_cloud']:
            if element in latest:
                ichimoku_data[element] = latest[element]
        
        # Fibonacci levels (if available)
        fibonacci_levels = {}
        for level in ['fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_0.786']:
            if level in latest:
                fibonacci_levels[level] = latest[level]
        
        # Support and resistance from recent candles
        recent_highs = df['high'].tail(20).nlargest(5).tolist()
        recent_lows = df['low'].tail(20).nsmallest(5).tolist()
        
        # Signal indicators
        signal_indicators = {}
        for signal in ['long_signal', 'short_signal', 'advanced_long_signal', 'advanced_short_signal']:
            if signal in latest:
                signal_indicators[signal] = latest[signal]
        
        # Comprehensive summary
        comprehensive_data = {
            'current_price': latest['close'],
            'price_action': {
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low'],
                'close': latest['close'],
                'price_change_5m': price_change_5m,
                'price_change_1h': price_change_1h,
                'price_change_24h': price_change_24h,
                'volatility': price_volatility,
                'candle_spread_pct': (latest['high'] - latest['low']) / latest['close'] * 100,
                'body_to_range_ratio': abs(latest['close'] - latest['open']) / (latest['high'] - latest['low']) if (latest['high'] - latest['low']) > 0 else 0,
            },
            'volume_metrics': volume_metrics,
            'indicator_values': indicator_values,
            'advanced_indicators': advanced_indicators,
            'ichimoku_data': ichimoku_data,
            'fibonacci_levels': fibonacci_levels,
            'price_patterns': price_patterns,
            'reversal_indicators': reversal_indicators,
            'recent_highs': recent_highs,
            'recent_lows': recent_lows,
            'signal_indicators': signal_indicators
        }
        
        return comprehensive_data
    
    def format_comprehensive_data(self, data: Dict) -> str:
        """Format the comprehensive data into a structured text for the LLM"""
        lines = []
        
        # Price action section
        lines.append("PRICE ACTION:")
        lines.append(f"Current price: ${data['current_price']:.2f}")
        lines.append(f"Open: ${data['price_action']['open']:.2f}")
        lines.append(f"High: ${data['price_action']['high']:.2f}")
        lines.append(f"Low: ${data['price_action']['low']:.2f}")
        lines.append(f"5m change: {data['price_action']['price_change_5m']:.2f}%")
        lines.append(f"1h change: {data['price_action']['price_change_1h']:.2f}%")
        lines.append(f"24h change: {data['price_action']['price_change_24h']:.2f}%")
        lines.append(f"Volatility: {data['price_action']['volatility']:.2f}%")
        lines.append(f"Candle spread: {data['price_action']['candle_spread_pct']:.2f}%")
        lines.append(f"Body/range ratio: {data['price_action']['body_to_range_ratio']:.2f}")
        
        # Technical indicator section
        lines.append("\nTECHNICAL INDICATORS:")
        indicators = data['indicator_values']
        lines.append(f"RSI: {indicators['rsi']:.2f}")
        lines.append(f"MACD: {indicators['macd']:.2f}")
        lines.append(f"MACD Signal: {indicators['macd_signal']:.2f}")
        lines.append(f"MACD Histogram: {indicators['macd_histogram']:.2f}")
        lines.append(f"SMA 20: ${indicators['sma_20']:.2f}")
        lines.append(f"SMA 50: ${indicators['sma_50']:.2f}")
        lines.append(f"SMA 200: ${indicators['sma_200']:.2f}")
        
        # Bollinger Bands
        lines.append(f"BB Upper: ${indicators['bb_upper']:.2f}")
        lines.append(f"BB Middle: ${indicators['bb_middle']:.2f}")
        lines.append(f"BB Lower: ${indicators['bb_lower']:.2f}")
        lines.append(f"ATR: {indicators['atr']:.2f}")
        
        # Advanced indicators (if available)
        if data['advanced_indicators']:
            lines.append("\nADVANCED INDICATORS:")
            for indicator, value in data['advanced_indicators'].items():
                if isinstance(value, (int, float)):
                    lines.append(f"{indicator.upper()}: {value:.2f}")
                else:
                    lines.append(f"{indicator.upper()}: {value}")
        
        # Ichimoku data (if available)
        if data['ichimoku_data']:
            lines.append("\nICHIMOKU CLOUD:")
            for indicator, value in data['ichimoku_data'].items():
                if isinstance(value, (int, float)):
                    lines.append(f"{indicator.replace('_', ' ').title()}: {value:.2f}")
                else:
                    lines.append(f"{indicator.replace('_', ' ').title()}: {value}")
        
        # Price patterns (if available)
        if data['price_patterns']:
            patterns_detected = [pattern for pattern, is_present in data['price_patterns'].items() if is_present]
            if patterns_detected:
                lines.append("\nPRICE PATTERNS DETECTED:")
                for pattern in patterns_detected:
                    lines.append(f"- {pattern.replace('_', ' ').title()}")
            else:
                lines.append("\nPRICE PATTERNS: None detected")
        
        # Reversal indicators (if available)
        if data['reversal_indicators']:
            lines.append("\nREVERSAL INDICATORS:")
            for indicator, value in data['reversal_indicators'].items():
                if isinstance(value, (int, float)):
                    lines.append(f"{indicator.replace('_', ' ').title()}: {value:.2f}")
                else:
                    lines.append(f"{indicator.replace('_', ' ').title()}: {value}")
        
        # Volume analysis
        lines.append("\nVOLUME ANALYSIS:")
        lines.append(f"Current volume: {data['volume_metrics']['current_volume']:.2f}")
        lines.append(f"Volume/SMA ratio: {data['volume_metrics']['volume_sma_ratio']:.2f}")
        
        # Support/Resistance levels
        lines.append("\nKEY PRICE LEVELS:")
        lines.append("Recent highs (resistance): " + ", ".join([f"${x:.2f}" for x in data['recent_highs']]))
        lines.append("Recent lows (support): " + ", ".join([f"${x:.2f}" for x in data['recent_lows']]))
        
        # Signal indicators
        lines.append("\nTECHNICAL SIGNALS:")
        for signal, value in data['signal_indicators'].items():
            lines.append(f"{signal.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines)
    
    def perform_multi_step_analysis(self, market_data: str, current_price: float, 
                                  timeframe: str, reversal_info: str = None,
                                  multi_timeframe_data: str = None) -> Dict:
        """
        Perform a multi-step analysis using the LLM
        
        Args:
            market_data: Formatted market data string
            current_price: Current price of the asset
            timeframe: Timeframe of the analysis
            reversal_info: Optional information about potential reversals
            multi_timeframe_data: Optional multi-timeframe analysis data
            
        Returns:
            Dictionary with LLM's multi-step analysis
        """
        try:
            # Create a model instance
            model = genai.GenerativeModel(model_name=self.model_name)
            
            # Step 1: Initial market assessment
            assessment_prompt = f"""
You are a professional cryptocurrency trader specializing in Bitcoin trading.

Analyze the following market data for BTC on the {timeframe} timeframe and provide an initial assessment:

{market_data}

{reversal_info if reversal_info else ""}

{multi_timeframe_data if multi_timeframe_data else ""}

Please provide your initial market assessment in the following JSON format:

```json
{{
"market_phase": "accumulation/distribution/markup/markdown/ranging",
"trend_strength": 0-100,
"momentum_direction": "bullish/bearish/neutral",
"key_levels": {{
    "strong_resistance": [list of prices],
    "weak_resistance": [list of prices],
    "strong_support": [list of prices],
    "weak_support": [list of prices]
}},
"overextended": true/false,
"reversal_likelihood": 0-100,
"consolidation_likelihood": 0-100,
"volatility_assessment": "high/medium/low"
}}
```
            """
            
            assessment_response = model.generate_content(assessment_prompt)
            assessment_text = assessment_response.text
            
            # Extract JSON from response
            assessment_json = self.extract_json_from_response(assessment_text)
            
            # Step 2: Technical analysis deep dive
            technical_prompt = f"""
Based on your initial assessment of the Bitcoin market data, perform a detailed technical analysis.

Your initial assessment was:
{json.dumps(assessment_json, indent=2)}

Focus especially on:
1. Potential reversals or continuation patterns
2. Volume confirmation or divergence
3. RSI, MACD, and Bollinger Band signals
4. Support/resistance interactions
5. The relationship between current price (${current_price:.2f}) and key levels

Please provide your technical analysis in the following JSON format:

```json
{{
"technical_signals": {{
    "rsi_signal": "overbought/oversold/neutral",
    "macd_signal": "bullish/bearish/neutral",
    "bollinger_signal": "upper_touch/lower_touch/middle_regression/outside_bands",
    "volume_confirmation": true/false,
    "reversal_signals": [list of any reversal signals detected],
    "continuation_signals": [list of any continuation signals detected]
}},
"price_action_analysis": "detailed analysis of recent price movements",
"key_indicator_divergences": [list of any divergences],
"probable_scenarios": [list of likely scenarios in order of probability]
}}
```
            """
            
            technical_response = model.generate_content(technical_prompt)
            technical_text = technical_response.text
            
            # Extract JSON from response
            technical_json = self.extract_json_from_response(technical_text)
            
            # Step 3: Generate final trading recommendations
            trading_prompt = f"""
Based on your comprehensive market analysis for Bitcoin at ${current_price:.2f} on the {timeframe} timeframe, provide a final trading recommendation.

Your market assessment was:
{json.dumps(assessment_json, indent=2)}

Your technical analysis was:
{json.dumps(technical_json, indent=2)}

Please provide your final trading recommendation in the following JSON format:

```json
{{
"market_condition": "bullish/bearish/neutral with brief explanation",
"signal": "LONG/SHORT/NEUTRAL",
"confidence": 0-100,
"support_levels": [list of prices],
"resistance_levels": [list of prices],
"stop_loss": price,
"take_profit": [list of target prices],
"risk_reward_ratio": number,
"reasoning": "brief explanation of your analysis",
"warning_signs": [list of things to watch that would invalidate this analysis]
}}
```
            """
            
            trading_response = model.generate_content(trading_prompt)
            trading_text = trading_response.text
            
            # Extract JSON from response
            trading_json = self.extract_json_from_response(trading_text)
            
            # Create combined analysis
            combined_analysis = {
                **trading_json,
                "market_assessment": assessment_json,
                "technical_analysis": technical_json
            }
            
            logger.info(f"Successfully performed multi-step LLM analysis: {combined_analysis['signal']} with {combined_analysis['confidence']}% confidence")
            return combined_analysis
        
        except Exception as e:
            logger.error(f"Error during multi-step LLM analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a basic fallback analysis
            fallback_analysis = {
                "market_condition": f"Error during LLM analysis: {str(e)}",
                "signal": "NEUTRAL",
                "confidence": 0,
                "support_levels": [],
                "resistance_levels": [],
                "stop_loss": current_price * 0.95,
                "take_profit": [current_price * 1.05],
                "risk_reward_ratio": 1.0,
                "reasoning": f"Analysis failed due to error: {str(e)}"
            }
            
            return fallback_analysis
    
    def extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON data from LLM response text"""
        try:
            # Find JSON block in response if it's wrapped in markdown code blocks
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response text: {response_text}")
            return {}
    
    def analyze_market_data(self, csv_data: str, 
                          current_price: float, 
                          timeframe: str,
                          multi_timeframe_data: str = None,
                          last_signal: str = None,
                          reversal_detected: bool = False,
                          reversal_data: Dict = None) -> Dict:
        """
        Send market data to LLM for multi-step analysis and receive trading signals
        
        Args:
            csv_data: CSV string of market data
            current_price: Current price of the asset
            timeframe: Timeframe of the analysis (e.g., '1h', '4h')
            multi_timeframe_data: Optional multi-timeframe analysis data
            last_signal: Previous trading signal (LONG/SHORT/NEUTRAL)
            reversal_detected: Whether a potential reversal was detected
            reversal_data: Dictionary with reversal detection information
            
        Returns:
            Dictionary with LLM's analysis and recommendations
        """
        try:
            # Convert CSV to DataFrame
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Prepare comprehensive data
            data_summary = self.prepare_comprehensive_data(df)
            
            # Format data in a more readable way
            formatted_summary = self.format_comprehensive_data(data_summary)
            
            # Prepare reversal information if detected
            reversal_info = None
            if reversal_detected and reversal_data:
                reversal_info = f"""
IMPORTANT REVERSAL POTENTIAL DETECTED:
Previous signal: {last_signal}
Reversal direction: {reversal_data.get('direction', 'Unknown')}
Reversal strength: {reversal_data.get('strength', 0)}/5
Reversal confidence: {reversal_data.get('confidence', 0)}%

The following reversal indicators were detected:
{reversal_data.get('indicators', [])}

Recent price action suggests a potential trend reversal. Please consider this carefully in your analysis.
"""
            
            # Perform multi-step analysis
            analysis = self.perform_multi_step_analysis(
                market_data=formatted_summary,
                current_price=current_price,
                timeframe=timeframe,
                reversal_info=reversal_info,
                multi_timeframe_data=multi_timeframe_data
            )
            
            return analysis
                
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a default structure if LLM call fails
            return {
                "market_condition": "unknown - LLM call failed",
                "signal": "NEUTRAL",
                "confidence": 0,
                "support_levels": [],
                "resistance_levels": [],
                "stop_loss": current_price * 0.95,
                "take_profit": [current_price * 1.05],
                "risk_reward_ratio": 1.0,
                "reasoning": f"LLM API error: {str(e)}"
            }