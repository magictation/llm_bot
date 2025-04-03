"""
Configuration module for the Bitcoin Trading Bot
"""
import os
import logging
import pathlib
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = pathlib.Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path)

# Setup logging
def setup_logging():
    """Configure logging for the application"""
    logs_dir = pathlib.Path(__file__).parent.parent.parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "trading_bot.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("BitcoinTradingBot")

# Create logger instance
logger = setup_logging()

def load_config() -> Dict[str, Any]:
    """
    Load configuration from .env file or environment variables
    
    Returns:
        Dictionary with configuration
    """
    config = {
        'symbol': os.getenv('TRADING_SYMBOL', 'BTCUSDT'),
        'timeframe': os.getenv('TRADING_TIMEFRAME', '15m'),
        'lookback_days': int(os.getenv('LOOKBACK_DAYS', '30')),
        'binance_api_key': os.getenv('BINANCE_API_KEY'),
        'binance_api_secret': os.getenv('BINANCE_API_SECRET'),
        'llm_api_key': os.getenv('GEMINI_API_KEY') or os.getenv('LLM_API_KEY')
    }
    
    # Log configuration (excluding sensitive data)
    logger.info(f"Loaded configuration: symbol={config['symbol']}, timeframe={config['timeframe']}")
    logger.info(f"Binance API configured: {bool(config['binance_api_key'] and config['binance_api_secret'])}")
    logger.info(f"LLM API configured: {bool(config['llm_api_key'])}")
    
    return config