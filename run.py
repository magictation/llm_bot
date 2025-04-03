#!/usr/bin/env python3
"""
Bitcoin Trading Signal Bot
Main entry point for running the bot or backtesting
"""
import os
import argparse
import pathlib
from dotenv import load_dotenv
from rich.console import Console
from app.bot.trading_bot import TradingBot
from app.config import logger

# Load environment variables from .env file in project root
dotenv_path = pathlib.Path(__file__).parent / '.env'
load_dotenv(dotenv_path)

console = Console()

def setup_cli():
    """Set up command line interface"""
    parser = argparse.ArgumentParser(description='Bitcoin Trading Signal Bot')
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the trading bot')
    run_parser.add_argument('--interval', type=int, default=15, help='Check interval in minutes')
    run_parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    run_parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    run_parser.add_argument('--env', default='.env', help='Path to .env file')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    backtest_parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    backtest_parser.add_argument('--env', default='.env', help='Path to .env file')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse command line arguments
    args = setup_cli()
    
    # Load custom .env file if specified
    if hasattr(args, 'env') and args.env != '.env':
        env_path = pathlib.Path(args.env)
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"Loaded environment from: {env_path}")
        else:
            logger.warning(f"Environment file not found: {env_path}")
    
    # Get API key for LLM from environment
    llm_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    
    # Bot configuration
    config = {
        'symbol': args.symbol if hasattr(args, 'symbol') else 'BTCUSDT',
        'timeframe': args.timeframe if hasattr(args, 'timeframe') else '1h',
        'lookback_days': int(os.getenv('LOOKBACK_DAYS', '30')),
        'llm_api_key': llm_api_key
    }
    
    # Initialize bot
    bot = TradingBot(config)
    
    # Execute command
    if args.command == 'run':
        interval = args.interval
        bot.run_continuous(interval_minutes=interval)
    elif args.command == 'backtest':
        bot.run_backtest(start_date=args.start, end_date=args.end)
    else:
        console.print("[bold red]Please specify a command: run or backtest[/bold red]")


if __name__ == "__main__":
    main()