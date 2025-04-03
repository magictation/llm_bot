"""
Main trading bot class for Bitcoin Trading Bot
"""
import os
import time
import json
import datetime
from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import schedule
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import logger
from app.data.binance_client import BinanceDataFetcher
from app.analysis.technical import TechnicalAnalyzer
from app.analysis.llm import LLMAnalyzer
from app.notification.alerts import NotificationManager

console = Console()

class TradingBot:
    """Main trading bot class that coordinates all components"""
    
    def __init__(self, config: Dict):
        """
        Initialize trading bot with configuration
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.symbol = config.get('symbol', 'BTCUSDT')
        self.timeframe = config.get('timeframe', '1h')
        self.lookback_days = config.get('lookback_days', 14)
        
        # Initialize components - using Binance for data only, no API keys needed
        self.data_fetcher = BinanceDataFetcher()
        
        if config.get('llm_api_key'):
            self.llm_analyzer = LLMAnalyzer(llm_api_key=config.get('llm_api_key'))
        else:
            self.llm_analyzer = None
            logger.warning("LLM Analyzer not initialized - missing API key")
        
        # Initialize state
        self.last_signal = "NEUTRAL"
        self.last_signal_price = 0
        self.last_signal_time = None
        
        logger.info(f"Trading Bot initialized for {self.symbol} on {self.timeframe} timeframe")
    
    def fetch_and_analyze(self) -> Tuple[pd.DataFrame, Dict, str]:
        """
        Fetch latest data, perform technical analysis, and get LLM insights
        
        Returns:
            Tuple of (analyzed_df, llm_analysis, signal)
        """
        # Calculate start date based on lookback days
        start_date = (datetime.datetime.now() - datetime.timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        # Fetch historical data
        df = self.data_fetcher.get_historical_klines(
            symbol=self.symbol,
            interval=self.timeframe,
            start_time=start_date
        )
        
        # Add technical indicators
        df = TechnicalAnalyzer.add_indicators(df)
        
        # Detect breakouts
        df = TechnicalAnalyzer.detect_breakouts(df)
        
        # Get current price
        current_price = self.data_fetcher.get_live_price(self.symbol)
        
        # Determine signal from technical indicators
        last_row = df.iloc[-1]
        if last_row['long_signal'] == 1:
            tech_signal = "LONG"
        elif last_row['short_signal'] == 1:
            tech_signal = "SHORT"
        else:
            tech_signal = "NEUTRAL"
        
        # Get LLM analysis if available
        if self.llm_analyzer:
            # Prepare data for LLM
            csv_data = TechnicalAnalyzer.prepare_data_for_llm(df)
            
            # Get LLM analysis
            llm_analysis = self.llm_analyzer.analyze_market_data(
                csv_data=csv_data,
                current_price=current_price,
                timeframe=self.timeframe
            )
        else:
            # Create a basic analysis without LLM
            llm_analysis = {
                "market_condition": "Unknown - LLM not available",
                "signal": tech_signal,
                "confidence": 70 if tech_signal != "NEUTRAL" else 0,
                "support_levels": [df['low'].tail(20).min() * 0.99],
                "resistance_levels": [df['high'].tail(20).max() * 1.01],
                "stop_loss": current_price * 0.95 if tech_signal == "LONG" else current_price * 1.05,
                "take_profit": [current_price * 1.05] if tech_signal == "LONG" else [current_price * 0.95],
                "risk_reward_ratio": 1.0,
                "reasoning": "Based on technical indicators only (LLM analysis not available)"
            }
        
        # Final signal determination (combining technical and LLM signals)
        if tech_signal != "NEUTRAL" and llm_analysis['signal'] == tech_signal:
            # Strong signal when both technical and LLM agree
            final_signal = tech_signal
        elif tech_signal != "NEUTRAL" and llm_analysis['confidence'] < 50:
            # Technical signal overrides low-confidence LLM
            final_signal = tech_signal
        elif llm_analysis['confidence'] > 70:
            # High-confidence LLM signal
            final_signal = llm_analysis['signal']
        else:
            # Default to NEUTRAL if no strong signal
            final_signal = "NEUTRAL"
        
        return df, llm_analysis, final_signal
    
    def check_for_signals(self):
        """Check for trading signals and generate alerts"""
        try:
            logger.info(f"Checking for signals on {self.symbol}...")
            
            # Fetch and analyze data
            df, llm_analysis, signal = self.fetch_and_analyze()
            
            # Get current price
            current_price = self.data_fetcher.get_live_price(self.symbol)
            
            # Check if signal has changed or if it's a significant update
            signal_changed = signal != self.last_signal
            high_confidence = llm_analysis['confidence'] > 80
            
            if signal_changed or (signal != "NEUTRAL" and high_confidence):
                # Update state
                self.last_signal = signal
                self.last_signal_price = current_price
                self.last_signal_time = datetime.datetime.now()
                
                # Log signal
                logger.info(f"Signal generated: {signal} at ${current_price:.2f}")
                
                # Send alert
                NotificationManager.console_alert(signal, current_price, llm_analysis)
                
                # Save signal data for reference
                self.save_signal_data(df, llm_analysis, signal, current_price)
            else:
                logger.info(f"No new signals. Current status: {signal} (Confidence: {llm_analysis['confidence']}%)")
                
        except Exception as e:
            logger.error(f"Error checking for signals: {e}")
    
    def save_signal_data(self, df: pd.DataFrame, analysis: Dict, signal: str, price: float):
        """
        Save signal data for later reference or backtesting
        
        Args:
            df: DataFrame with analyzed data
            analysis: LLM analysis dictionary
            signal: Signal type (LONG/SHORT/NEUTRAL)
            price: Current price when signal was generated
        """
        # Create signals directory if it doesn't exist
        os.makedirs('signals', exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save DataFrame to CSV
        df.tail(100).to_csv(f'signals/signal_data_{timestamp}.csv')
        
        # Save signal details to JSON
        signal_details = {
            'timestamp': timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal': signal,
            'price': price,
            'analysis': analysis
        }
        
        with open(f'signals/signal_details_{timestamp}.json', 'w') as f:
            json.dump(signal_details, f, indent=2)
            
        logger.info(f"Saved signal data to signals/signal_data_{timestamp}.csv")
    
    def run_continuous(self, interval_minutes: int = 15):
        """
        Run the bot continuously with a specified check interval
        
        Args:
            interval_minutes: Interval between checks in minutes
        """
        console.print(Panel.fit(
            "[bold blue]Bitcoin Trading Signal Bot[/bold blue]\n\n"
            f"[bold]Symbol:[/bold] {self.symbol}\n"
            f"[bold]Timeframe:[/bold] {self.timeframe}\n"
            f"[bold]Check Interval:[/bold] Every {interval_minutes} minutes\n",
            title="Starting Bot", border_style="green"
        ))
        
        # Schedule regular checks
        schedule.every(interval_minutes).minutes.do(self.check_for_signals)
        
        # Run initial check immediately
        self.check_for_signals()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before trying again
    
    def run_backtest(self, start_date: str, end_date: str = None):
        """
        Run a backtest over historical data
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
        """
        console.print(Panel.fit(
            "[bold blue]Bitcoin Trading Signal Backtest[/bold blue]\n\n"
            f"[bold]Symbol:[/bold] {self.symbol}\n"
            f"[bold]Timeframe:[/bold] {self.timeframe}\n"
            f"[bold]Period:[/bold] {start_date} to {end_date or 'today'}\n",
            title="Starting Backtest", border_style="yellow"
        ))
        
        if not end_date:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Fetch historical data
        df = self.data_fetcher.get_historical_klines(
            symbol=self.symbol,
            interval=self.timeframe,
            start_time=start_date,
            end_time=end_date
        )
        
        # Add technical indicators
        df = TechnicalAnalyzer.add_indicators(df)
        
        # Detect breakouts
        df = TechnicalAnalyzer.detect_breakouts(df)
        
        # Initialize backtest metrics
        trades = []
        current_position = None
        entry_price = 0
        
        # Iterate through data (skip first 50 rows for indicator warmup)
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            # Check for entry signals
            if current_position is None:  # Not in a position
                if row['long_signal'] == 1:
                    current_position = "LONG"
                    entry_price = row['close']
                    trades.append({
                        'entry_time': row.name,
                        'entry_price': entry_price,
                        'type': 'LONG',
                        'exit_time': None,
                        'exit_price': None,
                        'profit_pct': None
                    })
                elif row['short_signal'] == 1:
                    current_position = "SHORT"
                    entry_price = row['close']
                    trades.append({
                        'entry_time': row.name,
                        'entry_price': entry_price,
                        'type': 'SHORT',
                        'exit_time': None,
                        'exit_price': None,
                        'profit_pct': None
                    })
            
            # Check for exit conditions
            elif current_position == "LONG":
                # Simple exit strategy: exit when price falls below 20-day SMA or after 5 candles
                if row['close'] < row['sma_20'] or i - trades[-1]['entry_time'].value // 10**9 > 5 * df.index.to_series().diff().mean().total_seconds():
                    trades[-1]['exit_time'] = row.name
                    trades[-1]['exit_price'] = row['close']
                    trades[-1]['profit_pct'] = (row['close'] / entry_price - 1) * 100
                    current_position = None
            
            elif current_position == "SHORT":
                # Simple exit strategy: exit when price rises above 20-day SMA or after 5 candles
                if row['close'] > row['sma_20'] or i - trades[-1]['entry_time'].value // 10**9 > 5 * df.index.to_series().diff().mean().total_seconds():
                    trades[-1]['exit_time'] = row.name
                    trades[-1]['exit_price'] = row['close']
                    trades[-1]['profit_pct'] = (entry_price / row['close'] - 1) * 100
                    current_position = None
        
        # Close any open positions at the end of the backtest
        if current_position is not None:
            last_row = df.iloc[-1]
            trades[-1]['exit_time'] = last_row.name
            trades[-1]['exit_price'] = last_row['close']
            
            if current_position == "LONG":
                trades[-1]['profit_pct'] = (last_row['close'] / entry_price - 1) * 100
            else:  # SHORT
                trades[-1]['profit_pct'] = (entry_price / last_row['close'] - 1) * 100
        
        # Calculate backtest metrics
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            total_profit_pct = trades_df['profit_pct'].sum()
            win_rate = (trades_df['profit_pct'] > 0).mean() * 100
            avg_win = trades_df.loc[trades_df['profit_pct'] > 0, 'profit_pct'].mean() if len(trades_df.loc[trades_df['profit_pct'] > 0]) > 0 else 0
            avg_loss = trades_df.loc[trades_df['profit_pct'] < 0, 'profit_pct'].mean() if len(trades_df.loc[trades_df['profit_pct'] < 0]) > 0 else 0
            
            # Display results
            console.print(Panel.fit(
                f"[bold]Total Trades:[/bold] {len(trades_df)}\n"
                f"[bold]Win Rate:[/bold] {win_rate:.2f}%\n"
                f"[bold]Total Profit:[/bold] {total_profit_pct:.2f}%\n"
                f"[bold]Average Win:[/bold] {avg_win:.2f}%\n"
                f"[bold]Average Loss:[/bold] {avg_loss:.2f}%\n",
                title="Backtest Results", border_style="green"
            ))
            
            # Create a table of trades
            table = Table(title="Trade List")
            table.add_column("Entry Time", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Entry Price", style="green")
            table.add_column("Exit Time", style="cyan")
            table.add_column("Exit Price", style="red")
            table.add_column("Profit %", style="bold")
            
            for _, trade in trades_df.iterrows():
                profit_style = "green" if trade['profit_pct'] > 0 else "red"
                table.add_row(
                    str(trade['entry_time']),
                    trade['type'],
                    f"${trade['entry_price']:.2f}",
                    str(trade['exit_time']),
                    f"${trade['exit_price']:.2f}",
                    f"[{profit_style}]{trade['profit_pct']:.2f}%[/{profit_style}]"
                )
            
            console.print(table)
            
            # Plot equity curve
            trades_df['cumulative_profit'] = trades_df['profit_pct'].cumsum()
            plt.figure(figsize=(12, 6))
            plt.plot(trades_df.index, trades_df['cumulative_profit'])
            plt.title(f"Equity Curve: {self.symbol} {self.timeframe} ({start_date} to {end_date})")
            plt.xlabel("Trade Number")
            plt.ylabel("Cumulative Profit %")
            plt.grid(True)
            plt.savefig("backtest_equity_curve.png")
            plt.close()
            
            logger.info(f"Backtest completed: {len(trades_df)} trades, {total_profit_pct:.2f}% total profit")
            console.print(f"[bold green]Equity curve saved to[/bold green] backtest_equity_curve.png")
        else:
            console.print("[bold red]No trades were executed during the backtest period[/bold red]")