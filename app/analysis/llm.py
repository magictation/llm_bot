"""
LLM-based analysis module for Bitcoin Trading Bot
Optimized for gemma-3-27b-it model
"""
import os
import json
from typing import Dict
import pandas as pd
import google.generativeai as genai
from app.config import logger

class LLMAnalyzer:
    """Class to interact with LLM for trading insights"""
    
    def __init__(self, llm_api_key: str):
        """
        Initialize LLM analyzer with API key
        
        Args:
            llm_api_key: API key for the LLM service
        """
        # Clear any existing environment variable
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
            logger.info("Cleared existing GEMINI_API_KEY from environment")
            
        self.api_key = llm_api_key
        # Initialize Google AI client
        genai.configure(api_key=llm_api_key)
        # Use the gemma-3-27b-it model
        self.model_name = "gemma-3-27b-it"
        logger.info(f"Initialized LLM Analyzer with model: {self.model_name}")
    
    def prepare_summary_data(self, df):
        """Prepare a summary of the data for the LLM instead of raw CSV"""
        # Get the most recent data point
        latest = df.iloc[-1]
        
        # Calculate some key statistics
        price_change_24h = ((latest['close'] / df.iloc[-96]['close']) - 1) * 100 if len(df) > 96 else 0
        price_change_1h = ((latest['close'] / df.iloc[-4]['close']) - 1) * 100 if len(df) > 4 else 0
        
        # Create a summary
        summary = {
            'current_price': latest['close'],
            'open_price': latest['open'],
            'high_price': latest['high'],
            'low_price': latest['low'],
            'volume': latest['volume'],
            'price_change_1h': price_change_1h,
            'price_change_24h': price_change_24h,
            'rsi': latest.get('rsi', 0),
            'macd': latest.get('macd', 0),
            'macd_signal': latest.get('macd_signal', 0),
            'sma_20': latest.get('sma_20', 0),
            'sma_50': latest.get('sma_50', 0),
            'bb_upper': latest.get('bb_upper', 0),
            'bb_middle': latest.get('bb_middle', 0),
            'bb_lower': latest.get('bb_lower', 0),
            'long_signal': latest.get('long_signal', 0),
            'short_signal': latest.get('short_signal', 0)
        }
        
        # Get recent price levels for support/resistance
        last_10_highs = df['high'].tail(20).nlargest(5).tolist()
        last_10_lows = df['low'].tail(20).nsmallest(5).tolist()
        
        summary['recent_highs'] = last_10_highs
        summary['recent_lows'] = last_10_lows
        
        return summary
    
    def format_data_summary(self, data_summary):
        """Format the data summary in a readable way for the LLM"""
        return f"""
CURRENT MARKET DATA:
Current price: ${data_summary['current_price']}
1h change: {data_summary['price_change_1h']:.2f}%
24h change: {data_summary['price_change_24h']:.2f}%

TECHNICAL INDICATORS:
RSI: {data_summary['rsi']:.2f}
MACD: {data_summary['macd']:.2f}
MACD Signal: {data_summary['macd_signal']:.2f}
SMA 20: ${data_summary['sma_20']:.2f}
SMA 50: ${data_summary['sma_50']:.2f}

BOLLINGER BANDS:
Upper: ${data_summary['bb_upper']:.2f}
Middle: ${data_summary['bb_middle']:.2f}
Lower: ${data_summary['bb_lower']:.2f}

BREAKOUT SIGNALS:
Long signal triggered: {'Yes' if data_summary['long_signal'] == 1 else 'No'}
Short signal triggered: {'Yes' if data_summary['short_signal'] == 1 else 'No'}

RECENT PRICE LEVELS:
Recent highs: {', '.join([f"${x:.2f}" for x in data_summary['recent_highs']])}
Recent lows: {', '.join([f"${x:.2f}" for x in data_summary['recent_lows']])}
"""
    
    def analyze_market_data(self, csv_data: str, 
                          current_price: float, 
                          timeframe: str) -> Dict:
        """
        Send market data to LLM for analysis and receive trading signals
        
        Args:
            csv_data: CSV string of market data
            current_price: Current price of the asset
            timeframe: Timeframe of the analysis (e.g., '1h', '4h')
            
        Returns:
            Dictionary with LLM's analysis and recommendations
        """
        try:
            # Convert CSV to DataFrame
            df = pd.read_csv(pd.StringIO(csv_data))
            
            # Prepare summary data instead of using raw CSV
            data_summary = self.prepare_summary_data(df)
            
            # Format data in a more readable way
            formatted_summary = self.format_data_summary(data_summary)
            
            # Create an optimized prompt for gemma-3-27b-it
            prompt = f"""
You are a professional cryptocurrency trader analyzing Bitcoin price data for the {timeframe} timeframe.

Here is the current market data and technical indicators:

{formatted_summary}

Based on this data, please provide a trading analysis in the following JSON format:

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
  "reasoning": "brief explanation of your analysis"
}}
```

The JSON should only contain these fields and be properly formatted.
            """
            
            # Create a model instance with the current API
            model = genai.GenerativeModel(model_name=self.model_name)
            
            # Generate content using the updated API
            response = model.generate_content(prompt)
            
            # Extract text from response
            response_text = response.text
            
            # Find JSON block in response if it's wrapped in markdown code blocks
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            
            try:
                analysis = json.loads(json_str)
                logger.info(f"Successfully received LLM analysis: {analysis['signal']} with {analysis['confidence']}% confidence")
                return analysis
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {response_text}")
                # Return a default structure if JSON parsing fails
                return {
                    "market_condition": "unknown - LLM response parsing error",
                    "signal": "NEUTRAL",
                    "confidence": 0,
                    "support_levels": [],
                    "resistance_levels": [],
                    "stop_loss": current_price * 0.95,
                    "take_profit": [current_price * 1.05],
                    "risk_reward_ratio": 1.0,
                    "reasoning": "Failed to parse LLM response"
                }
                
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}")
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