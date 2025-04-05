#!/usr/bin/env python3
"""
Bitcoin Trading Signal Bot
Main entry point for running the bot or backtesting
"""
import os
import argparse
import pathlib
from app.backtesting.backtesting_engine import BacktestingEngine
from app.data.binance_client import BinanceDataFetcher
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from app.bot.trading_bot import TradingBot
from app.config import logger

# Load environment variables from .env file in project root
dotenv_path = pathlib.Path(__file__).parent / '.env'
load_dotenv(dotenv_path)

console = Console()

def setup_cli():
    """Set up enhanced command line interface"""
    parser = argparse.ArgumentParser(description='Bitcoin Trading Signal Bot')
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command with enhanced options
    run_parser = subparsers.add_parser('run', help='Run the trading bot')
    run_parser.add_argument('--interval', default='15m', help='Check interval (e.g., 30s, 5m, 1h)')
    run_parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    run_parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    run_parser.add_argument('--env', default='.env', help='Path to .env file')
    run_parser.add_argument('--advanced-indicators', action='store_true', help='Use advanced technical indicators')
    run_parser.add_argument('--multi-timeframe', action='store_true', help='Use multi-timeframe analysis')
    run_parser.add_argument('--timeframes', default='5m,15m,1h,4h', help='Comma-separated list of timeframes for multi-timeframe analysis')
    
    # Enhanced backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    backtest_parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    backtest_parser.add_argument('--env', default='.env', help='Path to .env file')
    backtest_parser.add_argument('--exit-strategy', default='composite', 
                                choices=['fixed_bars', 'trailing_stop', 'take_profit_stop_loss', 
                                        'moving_average', 'rsi', 'composite'],
                                help='Exit strategy to use')
    backtest_parser.add_argument('--position-size', type=float, default=100.0, 
                                help='Position size as percentage of capital')
    backtest_parser.add_argument('--advanced-indicators', action='store_true', 
                                help='Use advanced technical indicators')
    backtest_parser.add_argument('--tearsheet', action='store_true', 
                                help='Generate detailed tearsheet')
    
    # New walk-forward optimization command
    wfo_parser = subparsers.add_parser('walk-forward', help='Run walk-forward optimization')
    wfo_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    wfo_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    wfo_parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    wfo_parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    wfo_parser.add_argument('--windows', type=int, default=4, help='Number of test windows')
    wfo_parser.add_argument('--env', default='.env', help='Path to .env file')
    
    # New strategy comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare exit strategies')
    compare_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    compare_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    compare_parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    compare_parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    compare_parser.add_argument('--position-type', default='LONG', choices=['LONG', 'SHORT'], 
                               help='Position type to test')
    compare_parser.add_argument('--env', default='.env', help='Path to .env file')
    
    # New Monte Carlo simulation command
    monte_carlo_parser = subparsers.add_parser('monte-carlo', help='Run Monte Carlo simulation')
    monte_carlo_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    monte_carlo_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    monte_carlo_parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    monte_carlo_parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    monte_carlo_parser.add_argument('--simulations', type=int, default=1000, 
                                  help='Number of simulations')
    monte_carlo_parser.add_argument('--position-size', type=float, default=100.0, 
                                  help='Position size as percentage of capital')
    monte_carlo_parser.add_argument('--env', default='.env', help='Path to .env file')
    
    return parser.parse_args()

def parse_interval(interval_str):
    """
    Parse interval string like '1s', '5m', '1h' and convert to seconds
    
    Args:
        interval_str: String representing the interval (e.g., '30s', '5m', '1h')
        
    Returns:
        Interval in seconds
    """
    if not interval_str:
        return 15 * 60  # Default: 15 minutes in seconds
        
    # Remove any whitespace
    interval_str = str(interval_str).strip().lower()
    
    # If it's just a number, treat as minutes (backwards compatibility)
    if interval_str.isdigit():
        return int(interval_str) * 60
    
    # Match number and unit
    import re
    match = re.match(r'(\d+)([smh])', interval_str)
    
    if not match:
        raise ValueError(f"Invalid interval format: {interval_str}. Use format like '30s', '5m', '1h'")
    
    value, unit = match.groups()
    value = int(value)
    
    # Convert to seconds
    if unit == 's':
        return value
    elif unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    else:
        raise ValueError(f"Unknown time unit: {unit}. Use 's' for seconds, 'm' for minutes, 'h' for hours")


def main():
    """Enhanced main entry point"""
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
    
    # Common bot configuration
    config = {
        'symbol': args.symbol if hasattr(args, 'symbol') else 'BTCUSDT',
        'timeframe': args.timeframe if hasattr(args, 'timeframe') else '1h',
        'lookback_days': int(os.getenv('LOOKBACK_DAYS', '30')),
        'llm_api_key': llm_api_key
    }
    
    # Initialize data fetcher (used by both bot and backtesting)
    data_fetcher = BinanceDataFetcher()
    
    # Execute command
    if args.command == 'run':
        # Add advanced features to config if enabled
        if hasattr(args, 'advanced_indicators') and args.advanced_indicators:
            config['use_advanced_indicators'] = True
            
        if hasattr(args, 'multi_timeframe') and args.multi_timeframe:
            config['use_multi_timeframe'] = True
            config['timeframes'] = args.timeframes.split(',')
        
        # Process interval string to seconds
        interval_str = args.interval
        interval_seconds = parse_interval(interval_str)
        
        # Initialize bot
        bot = TradingBot(config)
        bot.run_continuous(interval_seconds=interval_seconds)
        
    elif args.command == 'backtest':
        # Initialize backtesting engine
        backtest_engine = BacktestingEngine(data_fetcher)
        
        # Determine technical analyzer to use
        if hasattr(args, 'advanced_indicators') and args.advanced_indicators:
            try:
                from app.analysis.advanced_technical import AdvancedTechnicalAnalyzer
                technical_analyzer = AdvancedTechnicalAnalyzer
                logger.info("Using advanced technical indicators for backtesting")
            except ImportError:
                from app.analysis.technical import TechnicalAnalyzer
                technical_analyzer = TechnicalAnalyzer
                logger.warning("Advanced technical indicators module not found, using standard indicators")
        else:
            from app.analysis.technical import TechnicalAnalyzer
            technical_analyzer = TechnicalAnalyzer
        
        # Prepare data
        df = backtest_engine.prepare_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            technical_analyzer=technical_analyzer
        )
        
        # Define exit params based on strategy
        exit_params = {}
        if args.exit_strategy == 'fixed_bars':
            exit_params = {'bars': 10}
        elif args.exit_strategy == 'trailing_stop':
            exit_params = {'atr_multiple': 2.0}
        elif args.exit_strategy == 'take_profit_stop_loss':
            exit_params = {'take_profit_pct': 5.0, 'stop_loss_pct': 2.0}
        elif args.exit_strategy == 'moving_average':
            exit_params = {'ma_period': 20}
        elif args.exit_strategy == 'rsi':
            exit_params = {'overbought': 70, 'oversold': 30}
        elif args.exit_strategy == 'composite':
            exit_params = {'atr_multiple': 2.0}
        
        # Extract exit parameters from environment if set (useful for parameter optimization)
        env_params = os.getenv('BACKTEST_PARAMS')
        if env_params:
            try:
                import ast
                env_params_dict = ast.literal_eval(env_params)
                exit_params.update(env_params_dict)
                logger.info(f"Using exit parameters from environment: {exit_params}")
            except:
                logger.warning(f"Failed to parse exit parameters from environment: {env_params}")
        
        # Run backtest for LONG positions
        console.print(Panel.fit(
            f"[bold]Running backtest for LONG positions...[/bold]",
            title="Backtest", border_style="yellow"
        ))
        
        # Determine signal column based on whether we're using advanced indicators
        if hasattr(args, 'advanced_indicators') and args.advanced_indicators and 'advanced_long_signal' in df.columns:
            long_signal_col = 'advanced_long_signal'
            short_signal_col = 'advanced_short_signal'
        else:
            long_signal_col = 'long_signal'
            short_signal_col = 'short_signal'
        
        result_long = backtest_engine.backtest_strategy(
            df=df,
            entry_signal_col=long_signal_col,
            position_type='LONG',
            exit_strategy=args.exit_strategy,
            exit_params=exit_params,
            position_size_pct=getattr(args, 'position_size', 100.0)
        )
        
        # Run backtest for SHORT positions
        console.print(Panel.fit(
            f"[bold]Running backtest for SHORT positions...[/bold]",
            title="Backtest", border_style="yellow"
        ))
        
        result_short = backtest_engine.backtest_strategy(
            df=df,
            entry_signal_col=short_signal_col,
            position_type='SHORT',
            exit_strategy=args.exit_strategy,
            exit_params=exit_params,
            position_size_pct=getattr(args, 'position_size', 100.0)
        )
        
        # Generate tearsheet if requested
        if hasattr(args, 'tearsheet') and args.tearsheet:
            console.print("[yellow]Generating tearsheets...[/yellow]")
            
            backtest_engine.generate_tearsheet(
                result_long, 
                args.symbol, 
                args.timeframe,
                strategy_name=f"LONG Strategy ({args.exit_strategy})"
            )
            
            backtest_engine.generate_tearsheet(
                result_short, 
                args.symbol, 
                args.timeframe,
                strategy_name=f"SHORT Strategy ({args.exit_strategy})"
            )
        
    elif args.command == 'walk-forward':
        # Initialize backtesting engine
        backtest_engine = BacktestingEngine(data_fetcher)
        
        # Run walk-forward optimization
        console.print(Panel.fit(
            f"[bold]Running walk-forward optimization...[/bold]\n"
            f"Symbol: {args.symbol}\n"
            f"Timeframe: {args.timeframe}\n"
            f"Period: {args.start} to {args.end or 'today'}\n"
            f"Windows: {args.windows}",
            title="Walk-Forward Optimization", border_style="blue"
        ))
        
        backtest_engine.walk_forward_optimization(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            test_windows=args.windows
        )
        
    elif args.command == 'compare':
        # Initialize backtesting engine
        backtest_engine = BacktestingEngine(data_fetcher)
        
        # Prepare data
        df = backtest_engine.prepare_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end
        )
        
        # Compare strategies
        console.print(Panel.fit(
            f"[bold]Comparing exit strategies for {args.position_type} positions...[/bold]\n"
            f"Symbol: {args.symbol}\n"
            f"Timeframe: {args.timeframe}\n"
            f"Period: {args.start} to {args.end or 'today'}",
            title="Strategy Comparison", border_style="magenta"
        ))
        
        backtest_engine.compare_strategies(
            df=df,
            entry_signal_col=f"{args.position_type.lower()}_signal",
            position_type=args.position_type
        )
        
    elif args.command == 'monte-carlo':
        # Initialize backtesting engine
        backtest_engine = BacktestingEngine(data_fetcher)
        
        # Prepare data
        df = backtest_engine.prepare_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end
        )
        
        # Run backtest to get trade data
        console.print("[yellow]Running backtest to get trade data...[/yellow]")
        
        # We need to use the external progress approach for consistency
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[yellow]{task.percentage:.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            backtest_task = progress.add_task("[cyan]Running backtest for trade data...", total=100)
            
            # Run backtest with the external progress
            result = backtest_engine.backtest_strategy(
                df=df,
                entry_signal_col='long_signal',
                position_type='LONG',
                exit_strategy='composite',
                external_progress=progress,
                task_id=backtest_task
            )
        
        if not result['success'] or len(result['trades']) < 10:
            console.print("[bold red]Not enough trades for meaningful Monte Carlo simulation.[/bold red]")
            console.print(f"[red]Found only {len(result['trades'])} trades. Need at least 10.[/red]")
            return
        
        # Run Monte Carlo simulation
        console.print(Panel.fit(
            f"[bold]Running Monte Carlo simulation...[/bold]\n"
            f"Symbol: {args.symbol}\n"
            f"Timeframe: {args.timeframe}\n"
            f"Simulations: {args.simulations}\n"
            f"Position Size: {args.position_size}%",
            title="Monte Carlo Simulation", border_style="cyan"
        ))
        
        backtest_engine.run_monte_carlo_simulation(
            trades_df=result['trades'],
            simulations=args.simulations,
            position_size_pct=args.position_size
        )
        
    else:
        console.print("[bold red]Please specify a command: run, backtest, walk-forward, compare, or monte-carlo[/bold red]")


if __name__ == "__main__":
    main()