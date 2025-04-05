"""
Improved backtesting framework for Bitcoin Trading Bot
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Callable, Optional, Union
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from app.config import logger
from app.analysis.technical import TechnicalAnalyzer

console = Console()

class BacktestingEngine:
    """Advanced backtesting engine for trading strategies"""
    
    def __init__(self, data_fetcher):
        """
        Initialize backtesting engine
        
        Args:
            data_fetcher: Instance of BinanceDataFetcher
        """
        self.data_fetcher = data_fetcher
        self.results_dir = os.path.join(os.getcwd(), 'backtest_results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def prepare_data(self, symbol: str, timeframe: str, 
                    start_date: str, end_date: str = None,
                    technical_analyzer = None) -> pd.DataFrame:
        """
        Prepare data for backtesting
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe for analysis (e.g., '1h', '4h')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            technical_analyzer: Technical analyzer class to use
            
        Returns:
            DataFrame with OHLCV data and indicators
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        console.print(f"[yellow]Fetching historical data for {symbol} ({timeframe}) from {start_date} to {end_date}...[/yellow]")
        
        # Fetch historical data
        df = self.data_fetcher.get_historical_klines(
            symbol=symbol,
            interval=timeframe,
            start_time=start_date,
            end_time=end_date
        )
        
        # Add technical indicators
        if technical_analyzer:
            # Check if we're using the AdvancedTechnicalAnalyzer
            if hasattr(technical_analyzer, 'add_all_indicators'):
                df = technical_analyzer.add_all_indicators(df)
                if hasattr(technical_analyzer, 'detect_advanced_breakouts'):
                    df = technical_analyzer.detect_advanced_breakouts(df)
                else:
                    # Fall back to standard breakout detection
                    from app.analysis.technical import TechnicalAnalyzer
                    df = TechnicalAnalyzer.detect_breakouts(df)
            else:
                # Standard TechnicalAnalyzer
                df = technical_analyzer.add_indicators(df)
                df = technical_analyzer.detect_breakouts(df)
        else:
            # No analyzer specified, use standard TechnicalAnalyzer
            from app.analysis.technical import TechnicalAnalyzer
            df = TechnicalAnalyzer.add_indicators(df)
            df = TechnicalAnalyzer.detect_breakouts(df)
            
        console.print(f"[green]Successfully prepared {len(df)} data points with indicators[/green]")
        
        return df
    
    def define_exit_strategies(self) -> Dict[str, Callable]:
        """
        Define exit strategies for backtesting
        
        Returns:
            Dictionary of strategy name to exit function
        """
        def fixed_bars_exit(df: pd.DataFrame, entry_idx: int, 
                           position_type: str, bars: int = 10) -> Tuple[int, float, str]:
            """Exit after a fixed number of bars"""
            exit_idx = min(entry_idx + bars, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            reason = f"Fixed {bars} bars exit"
            return exit_idx, exit_price, reason
        
        def trailing_stop_exit(df: pd.DataFrame, entry_idx: int, 
                              position_type: str, atr_multiple: float = 3.0) -> Tuple[int, float, str]:
            """Exit based on trailing stop using ATR"""
            entry_price = df.iloc[entry_idx]['close']
            
            # Calculate stop distance based on ATR
            atr = df.iloc[entry_idx]['atr']
            stop_distance = atr * atr_multiple
            
            # Initialize trailing values
            if position_type == "LONG":
                highest_high = entry_price
                stop_price = entry_price - stop_distance
            else:  # SHORT
                lowest_low = entry_price
                stop_price = entry_price + stop_distance
            
            # Loop through subsequent bars
            for i in range(entry_idx + 1, len(df)):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                # Check if stop was hit
                if position_type == "LONG" and current_low <= stop_price:
                    return i, stop_price, "Trailing stop hit"
                elif position_type == "SHORT" and current_high >= stop_price:
                    return i, stop_price, "Trailing stop hit"
                
                # Update trailing stop
                if position_type == "LONG" and current_high > highest_high:
                    highest_high = current_high
                    stop_price = highest_high - stop_distance
                elif position_type == "SHORT" and current_low < lowest_low:
                    lowest_low = current_low
                    stop_price = lowest_low + stop_distance
            
            # If we reach the end without hitting stop
            exit_idx = len(df) - 1
            exit_price = df.iloc[exit_idx]['close']
            reason = "End of data"
            return exit_idx, exit_price, reason
        
        def take_profit_stop_loss_exit(df: pd.DataFrame, entry_idx: int, 
                                      position_type: str, 
                                      take_profit_pct: float = 5.0,
                                      stop_loss_pct: float = 2.0) -> Tuple[int, float, str]:
            """Exit based on fixed take profit or stop loss percentages"""
            entry_price = df.iloc[entry_idx]['close']
            
            if position_type == "LONG":
                take_profit_price = entry_price * (1 + take_profit_pct / 100)
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            else:  # SHORT
                take_profit_price = entry_price * (1 - take_profit_pct / 100)
                stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
            
            # Loop through subsequent bars
            for i in range(entry_idx + 1, len(df)):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                # Check if take profit was hit
                if position_type == "LONG" and current_high >= take_profit_price:
                    return i, take_profit_price, "Take profit hit"
                elif position_type == "SHORT" and current_low <= take_profit_price:
                    return i, take_profit_price, "Take profit hit"
                
                # Check if stop loss was hit
                if position_type == "LONG" and current_low <= stop_loss_price:
                    return i, stop_loss_price, "Stop loss hit"
                elif position_type == "SHORT" and current_high >= stop_loss_price:
                    return i, stop_loss_price, "Stop loss hit"
            
            # If we reach the end without hitting TP or SL
            exit_idx = len(df) - 1
            exit_price = df.iloc[exit_idx]['close']
            reason = "End of data"
            return exit_idx, exit_price, reason
        
        def moving_average_exit(df: pd.DataFrame, entry_idx: int, 
                               position_type: str, ma_period: int = 20) -> Tuple[int, float, str]:
            """Exit when price crosses the moving average in the opposite direction"""
            # Calculate moving average column name
            ma_col = f'sma_{ma_period}'
            
            # Make sure the MA exists in the dataframe
            if ma_col not in df.columns:
                df[ma_col] = df['close'].rolling(window=ma_period).mean()
            
            # Loop through subsequent bars
            for i in range(entry_idx + 1, len(df)):
                current_close = df.iloc[i]['close']
                current_ma = df.iloc[i][ma_col]
                
                # Check for exit condition
                if position_type == "LONG" and current_close < current_ma:
                    return i, current_close, f"Price crossed below {ma_period}-period MA"
                elif position_type == "SHORT" and current_close > current_ma:
                    return i, current_close, f"Price crossed above {ma_period}-period MA"
            
            # If we reach the end without crossing MA
            exit_idx = len(df) - 1
            exit_price = df.iloc[exit_idx]['close']
            reason = "End of data"
            return exit_idx, exit_price, reason
        
        def rsi_exit(df: pd.DataFrame, entry_idx: int, 
                    position_type: str, 
                    overbought: float = 70, 
                    oversold: float = 30) -> Tuple[int, float, str]:
            """Exit when RSI reaches overbought/oversold levels"""
            # Loop through subsequent bars
            for i in range(entry_idx + 1, len(df)):
                current_rsi = df.iloc[i]['rsi']
                
                # Check for exit condition
                if position_type == "LONG" and current_rsi >= overbought:
                    return i, df.iloc[i]['close'], f"RSI reached overbought level ({overbought})"
                elif position_type == "SHORT" and current_rsi <= oversold:
                    return i, df.iloc[i]['close'], f"RSI reached oversold level ({oversold})"
            
            # If we reach the end without RSI condition
            exit_idx = len(df) - 1
            exit_price = df.iloc[exit_idx]['close']
            reason = "End of data"
            return exit_idx, exit_price, reason
        
        def composite_exit(df: pd.DataFrame, entry_idx: int, 
                          position_type: str,
                          atr_multiple: float = 2.0) -> Tuple[int, float, str]:
            """
            Composite exit strategy combining trailing stop with take profit
            
            Uses 3x ATR for take profit and trailing stop at 2x ATR
            """
            entry_price = df.iloc[entry_idx]['close']
            atr = df.iloc[entry_idx]['atr']
            
            # Set take profit and initial stop levels
            if position_type == "LONG":
                take_profit_price = entry_price + (3 * atr)
                stop_price = entry_price - (atr_multiple * atr)
                highest_high = entry_price
            else:  # SHORT
                take_profit_price = entry_price - (3 * atr)
                stop_price = entry_price + (atr_multiple * atr)
                lowest_low = entry_price
            
            # Loop through subsequent bars
            for i in range(entry_idx + 1, len(df)):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                # Check if take profit was hit
                if position_type == "LONG" and current_high >= take_profit_price:
                    return i, take_profit_price, "Take profit hit (3x ATR)"
                elif position_type == "SHORT" and current_low <= take_profit_price:
                    return i, take_profit_price, "Take profit hit (3x ATR)"
                
                # Check if stop was hit
                if position_type == "LONG" and current_low <= stop_price:
                    return i, stop_price, f"Trailing stop hit ({atr_multiple}x ATR)"
                elif position_type == "SHORT" and current_high >= stop_price:
                    return i, stop_price, f"Trailing stop hit ({atr_multiple}x ATR)"
                
                # Update trailing stop
                if position_type == "LONG" and current_high > highest_high:
                    highest_high = current_high
                    stop_price = max(stop_price, highest_high - (atr_multiple * atr))
                elif position_type == "SHORT" and current_low < lowest_low:
                    lowest_low = current_low
                    stop_price = min(stop_price, lowest_low + (atr_multiple * atr))
            
            # If we reach the end without hitting TP or stop
            exit_idx = len(df) - 1
            exit_price = df.iloc[exit_idx]['close']
            reason = "End of data"
            return exit_idx, exit_price, reason
        
        # Return dictionary of exit strategies
        return {
            "fixed_bars": fixed_bars_exit,
            "trailing_stop": trailing_stop_exit,
            "take_profit_stop_loss": take_profit_stop_loss_exit,
            "moving_average": moving_average_exit,
            "rsi": rsi_exit,
            "composite": composite_exit
        }
    
    def backtest_strategy(self, df: pd.DataFrame, 
                        entry_signal_col: str = 'long_signal', 
                        position_type: str = 'LONG',
                        exit_strategy: str = 'composite',
                        exit_params: Dict = None,
                        initial_capital: float = 1000.0,
                        position_size_pct: float = 100.0,
                        external_progress = None,
                        task_id = None) -> Dict:
        """
        Backtest a trading strategy
        
        Args:
            df: DataFrame with OHLCV data and indicators
            entry_signal_col: Column name for entry signals
            position_type: Type of position ('LONG' or 'SHORT')
            exit_strategy: Name of exit strategy to use
            exit_params: Parameters for exit strategy
            initial_capital: Initial capital for the backtest
            position_size_pct: Percentage of capital to use per trade
            external_progress: Optional external Progress object for nested progress bars
            task_id: Optional task ID when using external progress
            
        Returns:
            Dictionary with backtest results
        """
        # Define exit strategies
        exit_strategies = self.define_exit_strategies()
        
        # Validate inputs
        if entry_signal_col not in df.columns:
            raise ValueError(f"Entry signal column '{entry_signal_col}' not found in DataFrame")
            
        if position_type not in ['LONG', 'SHORT']:
            raise ValueError(f"Position type must be 'LONG' or 'SHORT', got '{position_type}'")
            
        if exit_strategy not in exit_strategies:
            raise ValueError(f"Exit strategy '{exit_strategy}' not found. Available strategies: {list(exit_strategies.keys())}")
            
        # Initialize default exit params if not provided
        if exit_params is None:
            exit_params = {}
            
        # Initialize backtest variables
        trades = []
        equity = [initial_capital]
        capital = initial_capital
        current_position = None
        entry_price = 0
        entry_idx = 0
        position_size = 0
        
        # Initialize bars list for time analysis
        bars_held = []
        
        # Get exit strategy function
        exit_func = exit_strategies[exit_strategy]
        
        # Only display this message if not using external progress
        if external_progress is None:
            console.print(f"[bold]Running backtest with {exit_strategy} exit strategy...[/bold]")
        
        # Determine whether to use external progress or create our own
        if external_progress is not None and task_id is not None:
            # Use external progress
            progress = external_progress
            task = task_id
            # Update description if it exists
            try:
                progress.update(task, description=f"Backtesting {position_type}...")
            except:
                pass
        else:
            # Create our own progress bar
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[yellow]{task.percentage:.0f}%"),
                TimeElapsedColumn(),
                console=console
            )
            
            # Start a new progress context only if we're creating our own
            progress_context = progress
            progress_context.__enter__()
            task = progress.add_task("[yellow]Backtesting...", total=len(df))
        
        try:
            # Loop through data (skip first 50 bars for indicator warmup)
            for i in range(50, len(df)):
                # Update progress
                if external_progress is not None:
                    # For external progress, only update occasionally to avoid slowdown
                    if i % 50 == 0 or i == len(df) - 1:
                        progress.update(task, completed=i, total=len(df))
                else:
                    progress.update(task, completed=i)
                
                row = df.iloc[i]
                
                # Update equity at each step if in a position
                if current_position:
                    current_price = row['close']
                    if current_position == "LONG":
                        current_value = position_size * (current_price / entry_price)
                    else:  # SHORT
                        current_value = position_size * (2 - current_price / entry_price)
                        
                    equity.append(capital - position_size + current_value)
                else:
                    equity.append(capital)
                
                # Check for entry signals
                if current_position is None:  # Not in a position
                    if position_type == 'LONG' and row[entry_signal_col] == 1:
                        current_position = "LONG"
                        entry_price = row['close']
                        entry_idx = i
                        position_size = capital * (position_size_pct / 100)
                        
                        trades.append({
                            'entry_time': row.name,
                            'entry_price': entry_price,
                            'type': 'LONG',
                            'exit_time': None,
                            'exit_price': None,
                            'profit_pct': None,
                            'profit_amount': None,
                            'exit_reason': None,
                            'bars_held': None
                        })
                        
                    elif position_type == 'SHORT' and row[entry_signal_col] == 1:
                        current_position = "SHORT"
                        entry_price = row['close']
                        entry_idx = i
                        position_size = capital * (position_size_pct / 100)
                        
                        trades.append({
                            'entry_time': row.name,
                            'entry_price': entry_price,
                            'type': 'SHORT',
                            'exit_time': None,
                            'exit_price': None,
                            'profit_pct': None,
                            'profit_amount': None,
                            'exit_reason': None,
                            'bars_held': None
                        })
                
                # Check for exit conditions
                elif current_position is not None:
                    # Apply exit strategy
                    exit_idx, exit_price, exit_reason = exit_func(
                        df, entry_idx, current_position, **exit_params
                    )
                    
                    # If exit condition is met
                    if i >= exit_idx:
                        # Calculate profit
                        if current_position == "LONG":
                            profit_pct = (exit_price / entry_price - 1) * 100
                            profit_amount = position_size * (exit_price / entry_price - 1)
                        else:  # SHORT
                            profit_pct = (entry_price / exit_price - 1) * 100
                            profit_amount = position_size * (entry_price / exit_price - 1)
                        
                        # Update trade record
                        trades[-1]['exit_time'] = df.iloc[exit_idx].name
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['profit_pct'] = profit_pct
                        trades[-1]['profit_amount'] = profit_amount
                        trades[-1]['exit_reason'] = exit_reason
                        trades[-1]['bars_held'] = exit_idx - entry_idx
                        
                        # Add to bars held list
                        bars_held.append(exit_idx - entry_idx)
                        
                        # Update capital
                        capital += profit_amount
                        
                        # Reset position
                        current_position = None
            
            # Close our own progress context if we created it
            if external_progress is None:
                progress_context.__exit__(None, None, None)
            
            # Create trade dataframe
            trades_df = pd.DataFrame(trades)
            
            if len(trades_df) == 0:
                return {
                    "success": False,
                    "message": "No trades were executed during the backtest period",
                    "trades": trades_df,
                    "equity_curve": pd.Series(equity),
                    "statistics": {}
                }
            
            # Calculate statistics
            statistics = self.calculate_statistics(trades_df, equity, initial_capital, bars_held)
            
            return {
                "success": True,
                "trades": trades_df,
                "equity_curve": pd.Series(equity),
                "statistics": statistics
            }
        
        except Exception as e:
            # Close our own progress context if we created it
            if external_progress is None:
                progress_context.__exit__(None, None, None)
            
            logger.error(f"Error during backtesting: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "message": f"Error during backtesting: {str(e)}",
                "trades": pd.DataFrame(),
                "equity_curve": pd.Series(equity),
                "statistics": {}
            }
    
    def calculate_statistics(self, trades_df: pd.DataFrame, 
                            equity: List[float], 
                            initial_capital: float,
                            bars_held: List[int]) -> Dict:
        """
        Calculate performance statistics with improved error handling
        
        Args:
            trades_df: DataFrame with trade records
            equity: List of equity values
            initial_capital: Initial capital
            bars_held: List of bars held per trade
            
        Returns:
            Dictionary with performance statistics
        """
        stats = {}
        
        try:
            # Basic trade statistics
            stats['total_trades'] = len(trades_df)
            
            if len(trades_df) == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_profit_amount': 0,
                    'total_profit_pct': 0,
                    'avg_winning_pct': 0,
                    'max_winning_pct': 0,
                    'avg_losing_pct': 0,
                    'max_losing_pct': 0,
                    'avg_trade_pct': 0,
                    'expectancy': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'current_drawdown': 0,
                    'sharpe_ratio': 0,
                    'calmar_ratio': 0,
                    'recovered_from_max_dd': False,
                    'recovery_bars': 0,
                    'avg_bars_held': 0,
                    'min_bars_held': 0,
                    'max_bars_held': 0
                }
            
            winning_trades = trades_df[trades_df['profit_pct'] > 0]
            losing_trades = trades_df[trades_df['profit_pct'] <= 0]
            
            stats['winning_trades'] = len(winning_trades)
            stats['losing_trades'] = len(losing_trades)
            
            stats['win_rate'] = len(winning_trades) / len(trades_df) * 100
            
            # Profit statistics
            stats['total_profit_amount'] = trades_df['profit_amount'].sum() if 'profit_amount' in trades_df.columns else trades_df['profit_pct'].sum()
            stats['total_profit_pct'] = (equity[-1] / initial_capital - 1) * 100
            
            if len(winning_trades) > 0:
                stats['avg_winning_pct'] = winning_trades['profit_pct'].mean()
                stats['max_winning_pct'] = winning_trades['profit_pct'].max()
            else:
                stats['avg_winning_pct'] = 0
                stats['max_winning_pct'] = 0
            
            if len(losing_trades) > 0:
                stats['avg_losing_pct'] = losing_trades['profit_pct'].mean()
                stats['max_losing_pct'] = losing_trades['profit_pct'].min()
            else:
                stats['avg_losing_pct'] = 0
                stats['max_losing_pct'] = 0
            
            stats['avg_trade_pct'] = trades_df['profit_pct'].mean()
            
            # Expectancy and risk metrics
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                avg_win = winning_trades['profit_pct'].mean()
                avg_loss = abs(losing_trades['profit_pct'].mean())
                win_rate = len(winning_trades) / len(trades_df)
                loss_rate = len(losing_trades) / len(trades_df)
                
                stats['expectancy'] = (win_rate * avg_win) - (loss_rate * avg_loss)
                loss_sum = losing_trades['profit_amount'].sum() if 'profit_amount' in losing_trades.columns else losing_trades['profit_pct'].sum()
                win_sum = winning_trades['profit_amount'].sum() if 'profit_amount' in winning_trades.columns else winning_trades['profit_pct'].sum()
                
                # Avoid division by zero
                if loss_sum != 0:
                    stats['profit_factor'] = abs(win_sum / loss_sum)
                else:
                    stats['profit_factor'] = float('inf') if win_sum > 0 else 0
            else:
                stats['expectancy'] = 0
                stats['profit_factor'] = 0
            
            # Drawdown calculation
            equity_series = pd.Series(equity)
            running_max = equity_series.cummax()
            drawdown = (equity_series / running_max - 1) * 100
            
            stats['max_drawdown'] = abs(drawdown.min())
            stats['current_drawdown'] = abs(drawdown.iloc[-1])
            
            # Calculate when max drawdown occurred
            max_dd_idx = drawdown.idxmin()
            
            # FIX: Avoid comparing datetime with integer
            try:
                # Convert index values to integers for comparison
                if hasattr(trades_df['entry_time'], 'dt'):
                    # If entry_time is already datetime
                    trades_df_idx = pd.Series(range(len(trades_df)))
                    trades_before_max_dd = len(trades_df[trades_df_idx < max_dd_idx])
                else:
                    # If we can directly compare numerically
                    trades_before_max_dd = len(trades_df[trades_df.index < max_dd_idx])
                
                stats['trades_to_max_drawdown'] = trades_before_max_dd
            except:
                # If comparison fails, use a simpler estimation
                stats['trades_to_max_drawdown'] = int(len(trades_df) * (max_dd_idx / len(equity_series)))
            
            # Sharpe Ratio (assuming risk-free rate of 0%)
            returns = [equity[i] / equity[i-1] - 1 for i in range(1, len(equity))]
            if len(returns) > 0 and np.std(returns) != 0:
                stats['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                stats['sharpe_ratio'] = 0
            
            # Calmar Ratio (annualized return / max drawdown)
            if stats['max_drawdown'] != 0:
                annualized_return = (equity[-1] / initial_capital) ** (252 / len(equity)) - 1
                stats['calmar_ratio'] = annualized_return / (stats['max_drawdown'] / 100)
            else:
                stats['calmar_ratio'] = 0
            
            # Recovery statistics
            if stats['max_drawdown'] > 0:
                # Find max drawdown point
                try:
                    peak_idx = running_max[max_dd_idx:].idxmax()
                    if peak_idx < len(equity_series) - 1:
                        # Check if recovered
                        recovered = False
                        recovery_bars = 0
                        for i in range(int(peak_idx) + 1, len(equity_series)):
                            recovery_bars += 1
                            if equity_series[i] >= running_max[peak_idx]:
                                recovered = True
                                break
                        
                        stats['recovered_from_max_dd'] = recovered
                        stats['recovery_bars'] = recovery_bars if recovered else None
                    else:
                        stats['recovered_from_max_dd'] = False
                        stats['recovery_bars'] = None
                except:
                    stats['recovered_from_max_dd'] = False
                    stats['recovery_bars'] = None
            else:
                stats['recovered_from_max_dd'] = True
                stats['recovery_bars'] = 0
            
            # Trade duration statistics
            if len(bars_held) > 0:
                stats['avg_bars_held'] = np.mean(bars_held)
                stats['min_bars_held'] = np.min(bars_held)
                stats['max_bars_held'] = np.max(bars_held)
            else:
                stats['avg_bars_held'] = 0
                stats['min_bars_held'] = 0
                stats['max_bars_held'] = 0
            
            # Monthly and yearly breakdown
            if 'entry_time' in trades_df.columns:
                try:
                    # Convert entry_time to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(trades_df['entry_time']):
                        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                    
                    trades_df['entry_month'] = trades_df['entry_time'].dt.strftime('%Y-%m')
                    trades_df['entry_year'] = trades_df['entry_time'].dt.year
                    
                    profit_col = 'profit_amount' if 'profit_amount' in trades_df.columns else 'profit_pct'
                    
                    monthly_returns = trades_df.groupby('entry_month')[profit_col].sum()
                    yearly_returns = trades_df.groupby('entry_year')[profit_col].sum()
                    
                    stats['monthly_returns'] = monthly_returns.to_dict()
                    stats['yearly_returns'] = {str(k): v for k, v in yearly_returns.to_dict().items()}
                    
                    # Best and worst months
                    if len(monthly_returns) > 0:
                        stats['best_month'] = monthly_returns.idxmax()
                        stats['best_month_return'] = monthly_returns.max()
                        stats['worst_month'] = monthly_returns.idxmin()
                        stats['worst_month_return'] = monthly_returns.min()
                except Exception as e:
                    logger.warning(f"Error calculating time-based statistics: {e}")
            
            # Consecutive wins and losses
            win_loss_streak = []
            current_streak = 0
            current_streak_type = None
            
            for _, trade in trades_df.iterrows():
                is_win = trade['profit_pct'] > 0
                
                if current_streak_type is None:
                    current_streak_type = is_win
                    current_streak = 1
                elif current_streak_type == is_win:
                    current_streak += 1
                else:
                    win_loss_streak.append((current_streak_type, current_streak))
                    current_streak_type = is_win
                    current_streak = 1
            
            # Add the last streak
            if current_streak_type is not None:
                win_loss_streak.append((current_streak_type, current_streak))
            
            # Calculate max consecutive wins and losses
            win_streaks = [streak for streak_type, streak in win_loss_streak if streak_type]
            loss_streaks = [streak for streak_type, streak in win_loss_streak if not streak_type]
            
            stats['max_consecutive_wins'] = max(win_streaks) if win_streaks else 0
            stats['max_consecutive_losses'] = max(loss_streaks) if loss_streaks else 0
        
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            import traceback
            traceback.print_exc()
            
            # Return basic stats if calculation fails
            return {
                'total_trades': len(trades_df),
                'winning_trades': len(trades_df[trades_df['profit_pct'] > 0]) if len(trades_df) > 0 else 0,
                'losing_trades': len(trades_df[trades_df['profit_pct'] <= 0]) if len(trades_df) > 0 else 0,
                'win_rate': len(trades_df[trades_df['profit_pct'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
                'total_profit_pct': (equity[-1] / initial_capital - 1) * 100,
                'error': str(e)
            }
        
        return stats
    
    def walk_forward_optimization(self, symbol: str, timeframe: str,
                                start_date: str, end_date: str,
                                test_windows: int = 4,
                                exit_strategies: List[str] = None,
                                exit_params_list: List[Dict] = None) -> Dict:
        """
        Perform walk-forward optimization
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe for analysis (e.g., '1h', '4h')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            test_windows: Number of test windows to divide the data into
            exit_strategies: List of exit strategies to test
            exit_params_list: List of exit parameters to test for each strategy
            
        Returns:
            Dictionary with walk-forward optimization results
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Default exit strategies if not provided
        if exit_strategies is None:
            exit_strategies = ['trailing_stop', 'take_profit_stop_loss', 'composite']
            
        # Default exit parameters if not provided
        if exit_params_list is None:
            exit_params_list = [
                {'atr_multiple': 2.0},  # For trailing_stop
                {'take_profit_pct': 5.0, 'stop_loss_pct': 2.0},  # For take_profit_stop_loss
                {'atr_multiple': 2.0}  # For composite
            ]
            
        # Fetch full dataset
        df = self.prepare_data(symbol, timeframe, start_date, end_date)
        
        # Calculate window sizes
        total_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
        window_days = total_days // test_windows
        
        # Initialize results
        window_results = []
        
        # Use a single Progress for all tasks
        with Progress(
            TextColumn("[bold blue]Walk-Forward Optimization"),
            BarColumn(),
            TextColumn("[yellow]{task.percentage:.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # Add tasks for windows and strategies
            window_task = progress.add_task("[yellow]Testing windows...", total=test_windows)
            strategy_task = progress.add_task("[green]Testing strategies...", total=len(exit_strategies), visible=False)
            backtest_task = progress.add_task("[cyan]Running backtest...", visible=False)
            
            # Loop through windows
            for i in range(test_windows):
                window_start = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * window_days)
                window_mid = window_start + timedelta(days=window_days // 2)
                window_end = window_start + timedelta(days=window_days)
                
                window_start_str = window_start.strftime('%Y-%m-%d')
                window_mid_str = window_mid.strftime('%Y-%m-%d')
                window_end_str = window_end.strftime('%Y-%m-%d')
                
                progress.update(window_task, description=f"[yellow]Window {i+1}/{test_windows}")
                console.print(f"[bold]Window {i+1}/{test_windows}[/bold]: {window_start_str} to {window_end_str}")
                
                # Get data for this window
                window_df = df[(df.index >= window_start) & (df.index <= window_end)]
                
                if len(window_df) < 100:
                    console.print(f"[red]Not enough data in window {i+1}. Skipping.[/red]")
                    progress.update(window_task, advance=1)
                    continue
                
                # Split into in-sample (training) and out-of-sample (testing)
                train_df = window_df[window_df.index < window_mid]
                test_df = window_df[window_df.index >= window_mid]
                
                console.print(f"Training data: {len(train_df)} bars, Testing data: {len(test_df)} bars")
                
                # Train on in-sample data to find best exit strategy
                best_strategy = None
                best_params = None
                best_expectancy = -float('inf')
                
                # Reset and show strategy task
                progress.update(strategy_task, completed=0, visible=True)
                
                for j, strategy in enumerate(exit_strategies):
                    progress.update(strategy_task, description=f"[green]Testing {strategy}...", completed=j)
                    
                    # Get parameters for this strategy
                    params = exit_params_list[j] if j < len(exit_params_list) else {}
                    
                    # Backtest on training data - using the progress for backtesting
                    progress.update(backtest_task, visible=True, completed=0)
                    result_long = self.backtest_strategy(
                        train_df, 
                        entry_signal_col='long_signal',
                        position_type='LONG',
                        exit_strategy=strategy,
                        exit_params=params,
                        external_progress=progress,
                        task_id=backtest_task
                    )
                    progress.update(backtest_task, visible=False)
                    
                    if not result_long['success'] or len(result_long['trades']) == 0:
                        continue
                        
                    expectancy = result_long['statistics'].get('expectancy', 0)
                    
                    if expectancy > best_expectancy:
                        best_expectancy = expectancy
                        best_strategy = strategy
                        best_params = params
                
                # Update strategy task
                progress.update(strategy_task, completed=len(exit_strategies), visible=False)
                
                # If no strategy worked well, use default
                if best_strategy is None:
                    console.print("[yellow]No good strategy found in training data, using default.[/yellow]")
                    best_strategy = 'composite'
                    best_params = {'atr_multiple': 2.0}
                
                console.print(f"[green]Best strategy: {best_strategy} with parameters {best_params}[/green]")
                
                # Test the best strategy on out-of-sample data
                progress.update(backtest_task, description="[cyan]Testing LONG strategy...", visible=True, completed=0)
                test_result_long = self.backtest_strategy(
                    test_df, 
                    entry_signal_col='long_signal',
                    position_type='LONG',
                    exit_strategy=best_strategy,
                    exit_params=best_params,
                    external_progress=progress,
                    task_id=backtest_task
                )
                
                progress.update(backtest_task, description="[cyan]Testing SHORT strategy...", completed=0)
                test_result_short = self.backtest_strategy(
                    test_df, 
                    entry_signal_col='short_signal',
                    position_type='SHORT',
                    exit_strategy=best_strategy,
                    exit_params=best_params,
                    external_progress=progress,
                    task_id=backtest_task
                )
                progress.update(backtest_task, visible=False)
                
                # Store results
                window_results.append({
                    'window': i + 1,
                    'start_date': window_start_str,
                    'mid_date': window_mid_str,
                    'end_date': window_end_str,
                    'best_strategy': best_strategy,
                    'best_params': best_params,
                    'train_expectancy': best_expectancy,
                    'test_long_trades': len(test_result_long['trades']),
                    'test_long_win_rate': test_result_long['statistics'].get('win_rate', 0),
                    'test_long_expectancy': test_result_long['statistics'].get('expectancy', 0),
                    'test_long_profit': test_result_long['statistics'].get('total_profit_pct', 0),
                    'test_short_trades': len(test_result_short['trades']),
                    'test_short_win_rate': test_result_short['statistics'].get('win_rate', 0),
                    'test_short_expectancy': test_result_short['statistics'].get('expectancy', 0),
                    'test_short_profit': test_result_short['statistics'].get('total_profit_pct', 0)
                })
                
                progress.update(window_task, advance=1)
        
        # Create summary of walk-forward results
        summary = pd.DataFrame(window_results)
        
        if len(summary) > 0:
            # Calculate averages for numeric columns only, excluding dictionaries/objects
            numeric_columns = [
                'test_long_trades', 'test_long_win_rate', 'test_long_expectancy', 'test_long_profit',
                'test_short_trades', 'test_short_win_rate', 'test_short_expectancy', 'test_short_profit'
            ]
            
            # Filter to only include columns that actually exist
            numeric_columns = [col for col in numeric_columns if col in summary.columns]
            
            # Calculate means only for numeric columns
            avg_results = summary[numeric_columns].mean()
            
            # Calculate robustness score (percentage of windows where test performance was positive)
            positive_long_windows = len(summary[summary['test_long_profit'] > 0])
            positive_short_windows = len(summary[summary['test_short_profit'] > 0])
            
            robustness_long = positive_long_windows / len(summary) * 100
            robustness_short = positive_short_windows / len(summary) * 100
        else:
            # Default values if no windows were analyzed
            avg_results = pd.Series({
                'test_long_win_rate': 0,
                'test_long_expectancy': 0,
                'test_long_profit': 0,
                'test_short_win_rate': 0,
                'test_short_expectancy': 0,
                'test_short_profit': 0
            })
            robustness_long = 0
            robustness_short = 0
        
        # Save results to file
        results_file = os.path.join(self.results_dir, f"walk_forward_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Convert avg_results to regular dict with float values
        avg_dict = {k: float(v) for k, v in avg_results.items()}
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'test_windows': test_windows,
            'exit_strategies': exit_strategies,
            'window_results': window_results,
            'average_results': avg_dict,
            'robustness_long': robustness_long,
            'robustness_short': robustness_short
        }
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Use default=str to handle serialization of dates and other objects
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        console.print(f"[bold green]Walk-forward optimization results saved to {results_file}[/bold green]")
        
        return results
    
    def compare_strategies(self, df: pd.DataFrame, entry_signal_col: str = 'long_signal',
                        position_type: str = 'LONG', initial_capital: float = 1000.0) -> Dict:
        """
        Compare different exit strategies on the same data
        
        Args:
            df: DataFrame with OHLCV data and indicators
            entry_signal_col: Column name for entry signals
            position_type: Type of position ('LONG' or 'SHORT')
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary with comparison results
        """
        # Define exit strategies
        exit_strategies = self.define_exit_strategies()
        
        # Dictionary to store results
        results = {}
        
        # Use a single Progress for all strategies
        with Progress(
            TextColumn("[bold blue]Comparing Strategies"),
            BarColumn(),
            TextColumn("[yellow]{task.percentage:.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # Add tasks for strategies and backtest
            strategy_task = progress.add_task("[yellow]Testing strategies...", total=len(exit_strategies))
            backtest_task = progress.add_task("[cyan]Running backtest...", visible=False)
            
            # Test each strategy
            for strategy_name, _ in exit_strategies.items():
                progress.update(strategy_task, description=f"Testing {strategy_name}")
                
                # Define parameters for this strategy
                exit_params = {}
                
                if strategy_name == 'fixed_bars':
                    exit_params = {'bars': 10}
                elif strategy_name == 'trailing_stop':
                    exit_params = {'atr_multiple': 2.0}
                elif strategy_name == 'take_profit_stop_loss':
                    exit_params = {'take_profit_pct': 5.0, 'stop_loss_pct': 2.0}
                elif strategy_name == 'moving_average':
                    exit_params = {'ma_period': 20}
                elif strategy_name == 'rsi':
                    exit_params = {'overbought': 70, 'oversold': 30}
                elif strategy_name == 'composite':
                    exit_params = {'atr_multiple': 2.0}
                
                # Run backtest
                progress.update(backtest_task, description=f"Running {strategy_name}...", visible=True, completed=0)
                result = self.backtest_strategy(
                    df, 
                    entry_signal_col=entry_signal_col,
                    position_type=position_type,
                    exit_strategy=strategy_name,
                    exit_params=exit_params,
                    initial_capital=initial_capital,
                    external_progress=progress,
                    task_id=backtest_task
                )
                progress.update(backtest_task, visible=False)
                
                # Store result
                results[strategy_name] = {
                    'success': result['success'],
                    'trades': len(result['trades']),
                    'win_rate': result['statistics'].get('win_rate', 0),
                    'expectancy': result['statistics'].get('expectancy', 0),
                    'profit_pct': result['statistics'].get('total_profit_pct', 0),
                    'max_drawdown': result['statistics'].get('max_drawdown', 0),
                    'sharpe_ratio': result['statistics'].get('sharpe_ratio', 0),
                    'profit_factor': result['statistics'].get('profit_factor', 0),
                    'avg_trade_pct': result['statistics'].get('avg_trade_pct', 0)
                }
                
                progress.update(strategy_task, advance=1)
        
        # Create comparison table
        table = Table(title=f"Strategy Comparison for {position_type} positions")
        
        table.add_column("Strategy", style="cyan")
        table.add_column("Trades", justify="right", style="white")
        table.add_column("Win Rate", justify="right", style="white")
        table.add_column("Profit %", justify="right", style="white")
        table.add_column("Expectancy", justify="right", style="white")
        table.add_column("Max DD", justify="right", style="white")
        table.add_column("Sharpe", justify="right", style="white")
        table.add_column("Profit Factor", justify="right", style="white")
        
        for strategy_name, stats in results.items():
            win_rate_color = "green" if stats['win_rate'] > 50 else "red"
            profit_color = "green" if stats['profit_pct'] > 0 else "red"
            
            table.add_row(
                strategy_name,
                str(stats['trades']),
                f"[{win_rate_color}]{stats['win_rate']:.2f}%[/{win_rate_color}]",
                f"[{profit_color}]{stats['profit_pct']:.2f}%[/{profit_color}]",
                f"{stats['expectancy']:.2f}",
                f"{stats['max_drawdown']:.2f}%",
                f"{stats['sharpe_ratio']:.2f}",
                f"{stats['profit_factor']:.2f}"
            )
        
        console.print(table)
        
        # Find best strategy
        if results:
            # Sort by different metrics
            best_profit = max(results.items(), key=lambda x: x[1]['profit_pct'])
            best_expectancy = max(results.items(), key=lambda x: x[1]['expectancy'])
            best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
            
            console.print(Panel(
                f"[bold]Best by Profit:[/bold] {best_profit[0]} ({best_profit[1]['profit_pct']:.2f}%)\n"
                f"[bold]Best by Expectancy:[/bold] {best_expectancy[0]} ({best_expectancy[1]['expectancy']:.2f})\n"
                f"[bold]Best by Sharpe Ratio:[/bold] {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})",
                title="Best Strategies", border_style="green"
            ))
        
        return results
    
    def generate_tearsheet(self, backtest_result: Dict, symbol: str, timeframe: str,
                          strategy_name: str = 'Default Strategy') -> None:
        """
        Generate a detailed backtest tearsheet
        
        Args:
            backtest_result: Dictionary with backtest results
            symbol: Trading pair symbol
            timeframe: Timeframe used
            strategy_name: Name of the strategy
        """
        if not backtest_result['success']:
            console.print("[red]Cannot generate tearsheet: Backtest was not successful[/red]")
            return
        
        trades_df = backtest_result['trades']
        equity_curve = backtest_result['equity_curve']
        stats = backtest_result['statistics']
        
        if len(trades_df) == 0:
            console.print("[red]Cannot generate tearsheet: No trades executed[/red]")
            return
        
        # Create result directory if not exists
        tearsheet_dir = os.path.join(self.results_dir, 'tearsheets')
        os.makedirs(tearsheet_dir, exist_ok=True)
        
        # Generate filename
        filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(tearsheet_dir, filename)
        
        # Create figure
        plt.figure(figsize=(15, 20))
        
        # Set style
        sns.set(style="whitegrid")
        
        # Create layout
        gs = plt.GridSpec(6, 2)
        
        # 1. Title
        plt.subplot(gs[0, :])
        plt.axis('off')
        plt.text(0.5, 0.5, f"Backtesting Tearsheet: {strategy_name}\n{symbol} {timeframe}", 
                horizontalalignment='center', verticalalignment='center', fontsize=16)
        
        # 2. Equity Curve
        plt.subplot(gs[1, :])
        plt.plot(equity_curve, color='blue')
        plt.title('Equity Curve')
        plt.grid(True)
        
        # 3. Drawdown
        plt.subplot(gs[2, 0])
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1) * 100
        plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown, color='red', linestyle='--')
        plt.title('Drawdown (%)')
        plt.grid(True)
        
        # 4. Monthly Returns
        plt.subplot(gs[2, 1])
        if 'monthly_returns' in stats:
            monthly_returns = pd.Series(stats['monthly_returns'])
            monthly_returns = monthly_returns.sort_index()
            colors = ['green' if x > 0 else 'red' for x in monthly_returns]
            plt.bar(range(len(monthly_returns)), monthly_returns.values, color=colors)
            plt.title('Monthly Returns')
            plt.xticks(range(len(monthly_returns)), monthly_returns.index, rotation=90)
            plt.grid(True)
        
        # 5. Trade Durations
        plt.subplot(gs[3, 0])
        trade_durations = trades_df['bars_held'].values
        plt.hist(trade_durations, bins=20, alpha=0.7, color='blue')
        plt.axvline(x=stats['avg_bars_held'], color='red', linestyle='--')
        plt.title(f'Trade Durations (Avg: {stats["avg_bars_held"]:.1f} bars)')
        plt.grid(True)
        
        # 6. Win/Loss Distribution
        plt.subplot(gs[3, 1])
        profit_pcts = trades_df['profit_pct'].values
        plt.hist(profit_pcts, bins=20, alpha=0.7, color='green')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Profit/Loss Distribution (%)')
        plt.grid(True)
        
        # 7. Trade Performance Over Time
        plt.subplot(gs[4, 0])
        trade_times = trades_df['entry_time'] if 'entry_time' in trades_df.columns else pd.Series(range(len(trades_df)))
        trade_profits = trades_df['profit_pct'].values
        colors = ['green' if x > 0 else 'red' for x in trade_profits]
        plt.scatter(range(len(trade_times)), trade_profits, color=colors)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.title('Trade Performance Over Time')
        plt.grid(True)
        
        # 8. Key Stats Table
        plt.subplot(gs[4, 1])
        plt.axis('off')
        
        key_stats = [
            f"Total Trades: {stats['total_trades']}",
            f"Win Rate: {stats['win_rate']:.2f}%",
            f"Total Profit: {stats['total_profit_pct']:.2f}%",
            f"Expectancy: {stats['expectancy']:.2f}",
            f"Profit Factor: {stats['profit_factor']:.2f}",
            f"Max Drawdown: {stats['max_drawdown']:.2f}%",
            f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}",
            f"Calmar Ratio: {stats['calmar_ratio']:.2f}",
            f"Avg Win: {stats['avg_winning_pct']:.2f}%",
            f"Avg Loss: {stats['avg_losing_pct']:.2f}%",
            f"Max Cons. Wins: {stats.get('max_consecutive_wins', 0)}",
            f"Max Cons. Losses: {stats.get('max_consecutive_losses', 0)}"
        ]
        
        plt.text(0.5, 0.5, '\n'.join(key_stats), 
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        
        # 9. Trade Timeline
        plt.subplot(gs[5, :])
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            for i, trade in trades_df.iterrows():
                is_win = trade['profit_pct'] > 0
                color = 'green' if is_win else 'red'
                plt.plot([trade['entry_time'], trade['exit_time']], [i, i], color=color, linewidth=2)
                plt.scatter(trade['entry_time'], i, color='blue')
                plt.scatter(trade['exit_time'], i, color=color)
            
            plt.title('Trade Timeline')
            plt.yticks(range(len(trades_df)), [f"Trade {i+1}" for i in range(len(trades_df))])
            plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        console.print(f"[bold green]Tearsheet saved to {filepath}[/bold green]")

    def run_monte_carlo_simulation(self, trades_df: pd.DataFrame, 
                                initial_capital: float = 1000.0,
                                simulations: int = 1000,
                                position_size_pct: float = 100.0) -> Dict:
        """
        Run Monte Carlo simulation to estimate strategy robustness
        
        Args:
            trades_df: DataFrame with trade records
            initial_capital: Initial capital
            simulations: Number of simulations to run
            position_size_pct: Percentage of capital to use per trade
            
        Returns:
            Dictionary with simulation results
        """
        if len(trades_df) == 0:
            console.print("[red]Cannot run simulation: No trades available[/red]")
            return {
                "success": False,
                "message": "No trades available for simulation"
            }
        
        # Extract profit percentages
        profit_pcts = trades_df['profit_pct'].values
        
        # Run simulations
        equity_curves = []
        max_drawdowns = []
        total_returns = []
        
        with Progress(
            TextColumn("[bold blue]Monte Carlo Simulation"),
            BarColumn(),
            TextColumn("[yellow]{task.percentage:.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[yellow]Running simulations...", total=simulations)
            
            for _ in range(simulations):
                # Shuffle the trades
                np.random.shuffle(profit_pcts)
                
                # Calculate equity curve
                equity = [initial_capital]
                capital = initial_capital
                
                for pct in profit_pcts:
                    position_size = capital * (position_size_pct / 100)
                    profit = position_size * (pct / 100)
                    capital += profit
                    equity.append(capital)
                
                equity_curves.append(equity)
                
                # Calculate max drawdown
                equity_series = pd.Series(equity)
                running_max = equity_series.cummax()
                drawdown = (equity_series / running_max - 1) * 100
                max_drawdowns.append(abs(drawdown.min()))
                
                # Calculate total return
                total_returns.append((equity[-1] / initial_capital - 1) * 100)
                
                progress.update(task, advance=1)
        
        # Convert to numpy arrays
        max_drawdowns = np.array(max_drawdowns)
        total_returns = np.array(total_returns)
        
        # Calculate statistics
        mean_return = total_returns.mean()
        median_return = np.median(total_returns)
        std_return = total_returns.std()
        
        mean_drawdown = max_drawdowns.mean()
        median_drawdown = np.median(max_drawdowns)
        worst_drawdown = max_drawdowns.max()
        
        # Calculate percentiles
        return_percentiles = {
            'p5': np.percentile(total_returns, 5),
            'p25': np.percentile(total_returns, 25),
            'p50': np.percentile(total_returns, 50),
            'p75': np.percentile(total_returns, 75),
            'p95': np.percentile(total_returns, 95)
        }
        
        drawdown_percentiles = {
            'p5': np.percentile(max_drawdowns, 5),
            'p25': np.percentile(max_drawdowns, 25),
            'p50': np.percentile(max_drawdowns, 50),
            'p75': np.percentile(max_drawdowns, 75),
            'p95': np.percentile(max_drawdowns, 95)
        }
        
        # Profit probability
        profit_probability = (total_returns > 0).mean() * 100
        
        # Calculating worst-case scenarios (95th percentile drawdown, 5th percentile return)
        worst_case_return = return_percentiles['p5']
        worst_case_drawdown = drawdown_percentiles['p95']
        
        # Calculate optimal position sizing based on worst case drawdown
        max_acceptable_drawdown = 30  # 30% max acceptable drawdown
        suggested_position_size = position_size_pct * (max_acceptable_drawdown / worst_case_drawdown)
        
        # Store all statistics
        simulation_results = {
            "success": True,
            "simulations": simulations,
            "initial_capital": initial_capital,
            "position_size_pct": position_size_pct,
            "mean_return": mean_return,
            "median_return": median_return,
            "std_return": std_return,
            "mean_drawdown": mean_drawdown,
            "median_drawdown": median_drawdown,
            "worst_drawdown": worst_drawdown,
            "return_percentiles": return_percentiles,
            "drawdown_percentiles": drawdown_percentiles,
            "profit_probability": profit_probability,
            "worst_case_return": worst_case_return,
            "worst_case_drawdown": worst_case_drawdown,
            "suggested_position_size": suggested_position_size
        }
        
        # Generate plots
        plt.figure(figsize=(15, 10))
        
        # Plot a sample of equity curves
        plt.subplot(2, 2, 1)
        sample_size = min(100, simulations)
        for i in range(sample_size):
            plt.plot(equity_curves[i], color='blue', alpha=0.1)
            
        # Add median equity curve
        median_curve = np.median(np.array(equity_curves), axis=0)
        plt.plot(median_curve, color='red', linewidth=2)
        
        plt.title('Sample of Equity Curves')
        plt.grid(True)
        
        # Plot return distribution
        plt.subplot(2, 2, 2)
        plt.hist(total_returns, bins=30, alpha=0.7, color='green')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.axvline(x=mean_return, color='black', linestyle='-', label=f'Mean: {mean_return:.2f}%')
        plt.axvline(x=return_percentiles['p5'], color='red', linestyle='-', label=f'5th %ile: {return_percentiles["p5"]:.2f}%')
        plt.axvline(x=return_percentiles['p95'], color='green', linestyle='-', label=f'95th %ile: {return_percentiles["p95"]:.2f}%')
        plt.legend()
        plt.title('Return Distribution')
        plt.grid(True)
        
        # Plot drawdown distribution
        plt.subplot(2, 2, 3)
        plt.hist(max_drawdowns, bins=30, alpha=0.7, color='red')
        plt.axvline(x=mean_drawdown, color='black', linestyle='-', label=f'Mean: {mean_drawdown:.2f}%')
        plt.axvline(x=drawdown_percentiles['p95'], color='red', linestyle='-', label=f'95th %ile: {drawdown_percentiles["p95"]:.2f}%')
        plt.legend()
        plt.title('Max Drawdown Distribution')
        plt.grid(True)
        
        # Plot return vs drawdown scatterplot
        plt.subplot(2, 2, 4)
        plt.scatter(max_drawdowns, total_returns, alpha=0.5)
        plt.xlabel('Max Drawdown (%)')
        plt.ylabel('Total Return (%)')
        plt.title('Return vs Drawdown')
        plt.grid(True)
        
        # Save figure
        monte_carlo_dir = os.path.join(self.results_dir, 'monte_carlo')
        os.makedirs(monte_carlo_dir, exist_ok=True)
        
        filename = f"monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(monte_carlo_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        # Print results
        console.print(Panel.fit(
            f"[bold]Monte Carlo Simulation Results[/bold]\n\n"
            f"Profit Probability: {profit_probability:.2f}%\n"
            f"Mean Return: {mean_return:.2f}% (std: {std_return:.2f}%)\n"
            f"Return Range (5th-95th): {return_percentiles['p5']:.2f}% to {return_percentiles['p95']:.2f}%\n\n"
            f"Mean Max Drawdown: {mean_drawdown:.2f}%\n"
            f"Worst-Case Drawdown (95th): {worst_case_drawdown:.2f}%\n\n"
            f"Suggested Position Size: {suggested_position_size:.2f}% of capital\n"
            f"(Based on limiting worst-case drawdown to 30%)",
            title="Simulation Summary", border_style="green"
        ))
        
        console.print(f"[bold green]Monte Carlo analysis plots saved to {filepath}[/bold green]")
        
        return simulation_results