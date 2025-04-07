#!/usr/bin/env python3
"""
Improved test script for LLM analysis with real market data.
This version is optimized for the gemma-3-27b-it model.
"""

import os
import sys
import json
import time
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from binance.client import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLM-Real-Data-Tester")

# Clear existing GEMINI_API_KEY
if "GEMINI_API_KEY" in os.environ:
    old_key = os.environ["GEMINI_API_KEY"]
    del os.environ["GEMINI_API_KEY"]
    logger.info(f"Deleted old GEMINI_API_KEY from environment: {old_key[:5]}...{old_key[-4:]}")

# Load from .env file
script_dir = Path(__file__).parent
env_path = script_dir / '.env'

if env_path.exists():
    load_dotenv(env_path, override=True)
    logger.info(f"Loaded environment from: {env_path}")
else:
    logger.error("No .env file found. Please create one with your GEMINI_API_KEY.")
    sys.exit(1)

# Verify API key
new_key = os.getenv("GEMINI_API_KEY")
if not new_key:
    logger.error("No GEMINI_API_KEY found in .env file!")
    sys.exit(1)
logger.info(f"Using GEMINI_API_KEY from .env: {new_key[:5]}...{new_key[-4:]}")

def add_technical_indicators(df):
    """Add technical indicators to DataFrame"""
    df = df.copy()
    
    # Calculate moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, 0.00001)
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Add breakout signals
    df['long_signal'] = ((df['close'] > df['bb_upper'].shift(1)) & 
                         (df['volume'] > df['volume'].rolling(20).mean() * 1.5)).astype(int)
    df['short_signal'] = ((df['close'] < df['bb_lower'].shift(1)) & 
                          (df['volume'] > df['volume'].rolling(20).mean() * 1.5)).astype(int)
    
    return df.dropna()

def get_binance_data(symbol='BTCUSDT', interval='15m', limit=100):
    """Fetch real market data from Binance"""
    try:
        client = Client()
        logger.info(f"Fetching {limit} {interval} candles for {symbol} from Binance...")
        
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} candles")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from Binance: {e}")
        raise

def prepare_summary_data(df):
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

def test_with_real_data():
    """Test LLM analysis with real market data"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("No GEMINI_API_KEY found in environment variables")
        return False
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Get real market data
        df = get_binance_data(interval='15m', limit=100)
        
        # Get current price
        current_price = float(df['close'].iloc[-1])
        logger.info(f"Current BTC price: ${current_price}")
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Prepare data summary for LLM
        data_summary = prepare_summary_data(df)
        
        # Format the summary in a cleaner way
        formatted_summary = f"""
CURRENT MARKET DATA (BTC/USDT 15m):
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
        
        timeframe = '15m'
        
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
        
        logger.info("Sending real market data analysis request to LLM...")
        start_time = time.time()
        
        # Create the model - Using gemma-3-27b-it model
        # model = genai.GenerativeModel(model_name="gemma-3-27b-it")
        # model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Extract text from response
        response_text = response.text
        end_time = time.time()
        
        logger.info(f"Received response in {end_time - start_time:.2f} seconds")
        
        # Extract JSON from response
        json_str = response_text
        
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        
        # Try to parse JSON
        try:
            analysis = json.loads(json_str)
            
            # Print analysis summary
            logger.info(f"\n{'=' * 50}\nLLM ANALYSIS RESULTS\n{'=' * 50}")
            logger.info(f"Current BTC Price: ${current_price}")
            logger.info(f"Market condition: {analysis.get('market_condition', 'N/A')}")
            logger.info(f"Signal: {analysis.get('signal', 'N/A')} with {analysis.get('confidence', 'N/A')}% confidence")
            logger.info(f"Support levels: {analysis.get('support_levels', 'N/A')}")
            logger.info(f"Resistance levels: {analysis.get('resistance_levels', 'N/A')}")
            logger.info(f"Stop loss: ${analysis.get('stop_loss', 'N/A')}")
            logger.info(f"Take profit targets: {analysis.get('take_profit', 'N/A')}")
            logger.info(f"Risk/reward ratio: {analysis.get('risk_reward_ratio', 'N/A')}")
            logger.info(f"\nReasoning:\n{analysis.get('reasoning', 'N/A')}")
            
            logger.info("\n🎉 Successfully tested LLM with real market data!")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw response: {response_text}")
            return False
            
    except Exception as e:
        logger.error(f"Error during real data test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting LLM test with real market data")
    success = test_with_real_data()
    sys.exit(0 if success else 1)