"""
Configuration module for the Bitcoin Trading Bot
"""
import os
import logging
import pathlib
from typing import Dict, Any
from dotenv import load_dotenv

# Determine the project root directory
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Load environment variables from .env file
def find_and_load_dotenv():
    """
    Find and load .env file from the project root or parent directories
    """
    # List of possible .env file locations
    possible_env_files = [
        PROJECT_ROOT / '.env',
        PROJECT_ROOT.parent / '.env',
        pathlib.Path.home() / '.env',
        PROJECT_ROOT / '.env.local',
        PROJECT_ROOT / '.env.example'
    ]
    
    for env_path in possible_env_files:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"Loaded environment from: {env_path}")
            return
    
    print("No .env file found. Using system environment variables.")

# Find and load .env file
find_and_load_dotenv()

# Setup logging
def setup_logging():
    """Configure logging for the application"""
    logs_dir = PROJECT_ROOT / 'logs'
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
    # Explicitly print out environment variables for debugging
    logger.info("Checking environment variables:")
    logger.info(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY', 'Not set')}")
    logger.info(f"LLM_API_KEY: {os.getenv('LLM_API_KEY', 'Not set')}")
    
    config = {
        'symbol': os.getenv('TRADING_SYMBOL', 'BTCUSDT'),
        'timeframe': os.getenv('TRADING_TIMEFRAME', '15m'),
        'lookback_days': int(os.getenv('LOOKBACK_DAYS', '30')),
        'binance_api_key': os.getenv('BINANCE_API_KEY'),
        'binance_api_secret': os.getenv('BINANCE_API_SECRET'),
        # Prioritize GEMINI_API_KEY, then fall back to LLM_API_KEY
        'llm_api_key': os.getenv('GEMINI_API_KEY') or os.getenv('LLM_API_KEY')
    }
    
    # Log configuration (excluding sensitive data)
    logger.info(f"Loaded configuration: symbol={config['symbol']}, timeframe={config['timeframe']}")
    logger.info(f"Binance API configured: {bool(config['binance_api_key'] and config['binance_api_secret'])}")
    logger.info(f"LLM API configured: {bool(config['llm_api_key'])}")
    
    # Raise an error if no LLM API key is found
    if not config['llm_api_key']:
        logger.error("No LLM API key found. Please set GEMINI_API_KEY or LLM_API_KEY.")
        raise ValueError("No LLM API key provided. Please check your .env file or environment variables.")
    
    return config