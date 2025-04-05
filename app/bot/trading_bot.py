"""
Main trading bot class for Bitcoin Trading Bot
"""
import os
import time
import json
import datetime
from typing import Dict, Tuple, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import schedule
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import logger
from app.data.binance_client import BinanceDataFetcher
from app.analysis.technical import TechnicalAnalyzer
from app.notification.alerts import NotificationManager

# These imports will only be used conditionally, so we don't need to have them at the top level
# They'll be imported within the methods that use them when those features are enabled

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
        
        # Initialize advanced components if enabled
        self.use_advanced_indicators = config.get('use_advanced_indicators', False)
        self.use_multi_timeframe = config.get('use_multi_timeframe', False)
        
        if self.use_multi_timeframe:
            timeframes = config.get('timeframes', ['5m', '15m', '1h', '4h'])
            from app.analysis.multi_timeframe import MultiTimeframeAnalyzer
            self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(self.data_fetcher, timeframes)
            logger.info(f"Initialized Multi-Timeframe Analyzer with timeframes: {timeframes}")
        else:
            self.multi_timeframe_analyzer = None
        
        if config.get('llm_api_key'):
            from app.analysis.llm import LLMAnalyzer
            self.llm_analyzer = LLMAnalyzer(llm_api_key=config.get('llm_api_key'))
        else:
            self.llm_analyzer = None
            logger.warning("LLM Analyzer not initialized - missing API key")
        
        # Initialize state
        self.last_signal = "NEUTRAL"
        self.last_signal_price = 0
        self.last_signal_time = None
        
        logger.info(f"Trading Bot initialized for {self.symbol} on {self.timeframe} timeframe")
        if self.use_advanced_indicators:
            logger.info("Using advanced technical indicators")
        if self.use_multi_timeframe:
            logger.info("Using multi-timeframe analysis")
    
    def fetch_and_analyze(self) -> Tuple[pd.DataFrame, Dict, str]:
        """
        Fetch latest data, perform technical analysis, and get LLM insights
        
        Returns:
            Tuple of (analyzed_df, llm_analysis, signal)
        """
        # Calculate start date based on lookback days
        start_date = (datetime.datetime.now() - datetime.timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        # Fetch historical data for primary timeframe
        df = self.data_fetcher.get_historical_klines(
            symbol=self.symbol,
            interval=self.timeframe,
            start_time=start_date
        )
        
        # Add technical indicators based on configuration
        if self.use_advanced_indicators:
            try:
                from app.analysis.advanced_technical import AdvancedTechnicalAnalyzer
                df = AdvancedTechnicalAnalyzer.add_all_indicators(df)
                df = AdvancedTechnicalAnalyzer.detect_advanced_breakouts(df)
                logger.info("Applied advanced technical indicators")
            except Exception as e:
                logger.error(f"Error applying advanced indicators, falling back to standard: {e}")
                df = TechnicalAnalyzer.add_indicators(df)
                df = TechnicalAnalyzer.detect_breakouts(df)
        else:
            df = TechnicalAnalyzer.add_indicators(df)
            df = TechnicalAnalyzer.detect_breakouts(df)
        
        # Get current price
        current_price = self.data_fetcher.get_live_price(self.symbol)
        
        # Get multi-timeframe analysis if enabled
        multi_timeframe_analysis = None
        if self.use_multi_timeframe and self.multi_timeframe_analyzer:
            try:
                _, multi_timeframe_analysis = self.multi_timeframe_analyzer.get_multi_timeframe_analysis(
                    self.symbol, self.lookback_days
                )
                logger.info(f"Multi-timeframe analysis signal: {multi_timeframe_analysis.get('signal', 'N/A')} "
                          f"with {multi_timeframe_analysis.get('confidence', 0):.2f}% confidence")
            except Exception as e:
                logger.error(f"Error performing multi-timeframe analysis: {e}")
                multi_timeframe_analysis = None
        
        # Determine signal from technical indicators
        last_row = df.iloc[-1]
        
        if self.use_advanced_indicators and 'advanced_long_signal' in df.columns:
            if last_row['advanced_long_signal'] == 1:
                tech_signal = "LONG"
            elif last_row['advanced_short_signal'] == 1:
                tech_signal = "SHORT"
            else:
                tech_signal = "NEUTRAL"
        else:
            if last_row['long_signal'] == 1:
                tech_signal = "LONG"
            elif last_row['short_signal'] == 1:
                tech_signal = "SHORT"
            else:
                tech_signal = "NEUTRAL"
        
        # Get LLM analysis if available
        if self.llm_analyzer:
            try:
                # Prepare data for LLM
                if self.use_advanced_indicators:
                    try:
                        from app.analysis.advanced_technical import AdvancedTechnicalAnalyzer
                        csv_data = AdvancedTechnicalAnalyzer.prepare_advanced_data_for_llm(df)
                    except Exception as e:
                        logger.error(f"Error preparing advanced data for LLM: {e}")
                        csv_data = TechnicalAnalyzer.prepare_data_for_llm(df)
                else:
                    csv_data = TechnicalAnalyzer.prepare_data_for_llm(df)

                # Add multi-timeframe data if available
                multi_timeframe_summary = ""
                if self.use_multi_timeframe and multi_timeframe_analysis and self.multi_timeframe_analyzer:
                    try:
                        multi_timeframe_summary = self.multi_timeframe_analyzer.prepare_multi_timeframe_summary_for_llm(
                            multi_timeframe_analysis
                        )
                        logger.info("Added multi-timeframe analysis to LLM input")
                    except Exception as e:
                        logger.error(f"Error preparing multi-timeframe summary for LLM: {e}")
                        multi_timeframe_summary = ""

                # Get LLM analysis with combined data
                llm_analysis = self.llm_analyzer.analyze_market_data(
                    csv_data=csv_data,
                    current_price=current_price,
                    timeframe=self.timeframe,
                    multi_timeframe_data=multi_timeframe_summary if multi_timeframe_summary else None
                )
            except Exception as e:
                logger.error(f"Error getting LLM analysis: {e}")
                # Create a basic analysis as fallback
                llm_analysis = {
                    "market_condition": "Error retrieving LLM analysis",
                    "signal": tech_signal,
                    "confidence": 50 if tech_signal != "NEUTRAL" else 0,
                    "support_levels": [df['low'].tail(20).min() * 0.99],
                    "resistance_levels": [df['high'].tail(20).max() * 1.01],
                    "stop_loss": current_price * 0.95 if tech_signal == "LONG" else current_price * 1.05,
                    "take_profit": [current_price * 1.05] if tech_signal == "LONG" else [current_price * 0.95],
                    "risk_reward_ratio": 1.0,
                    "reasoning": f"Based on technical indicators (LLM analysis failed: {e})"
                }
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
        
        # Final signal determination
        if self.use_multi_timeframe and multi_timeframe_analysis:
            # Use multi-timeframe analysis as a filter
            mt_signal = multi_timeframe_analysis.get('signal', 'NEUTRAL')
            mt_confidence = multi_timeframe_analysis.get('confidence', 0)
            
            if tech_signal != "NEUTRAL" and mt_signal == tech_signal:
                # Strong signal when both technical and multi-timeframe agree
                final_signal = tech_signal
                # Increase LLM confidence when all three agree
                if llm_analysis['signal'] == tech_signal:
                    llm_analysis['confidence'] = min(100, llm_analysis['confidence'] * 1.2)
            elif tech_signal != "NEUTRAL" and mt_confidence < 60:
                # Technical signal overrides low-confidence multi-timeframe
                final_signal = tech_signal
            elif mt_confidence > 80:
                # High-confidence multi-timeframe signal
                final_signal = mt_signal
            elif llm_analysis['confidence'] > 70:
                # High-confidence LLM signal
                final_signal = llm_analysis['signal']
            else:
                # Default to NEUTRAL if no strong signal
                final_signal = "NEUTRAL"
        else:
            # Original logic without multi-timeframe
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
            
            # Enhanced signal processing
            confidence = llm_analysis.get('confidence', 0)
            llm_signal = llm_analysis.get('signal', 'NEUTRAL')
            
            logger.info(f"LLM Analysis: Signal={llm_signal}, Confidence={confidence}%")
            
            # Modify the signal processing logic
            is_significant_signal = (
                confidence >= 50 and  # Confidence threshold
                (llm_signal in ["LONG", "SHORT"])  # Meaningful signal
            )
            
            # Send alert if signal is significant
            if is_significant_signal:
                # Initialize notification manager
                notification_manager = NotificationManager()
                
                # Log the signal details
                logger.info(f"Sending notifications for signal: {llm_signal} at ${current_price:.2f}")
                
                # Send console alert
                notification_manager.console_alert(llm_signal, current_price, llm_analysis)
                
                # Send Telegram alert
                notification_manager.telegram_alert(llm_signal, current_price, llm_analysis)
                
                # Update last signal
                self.last_signal = llm_signal
                self.last_signal_price = current_price
                self.last_signal_time = datetime.datetime.now()
                
                # Save signal data for reference
                self.save_signal_data(df, llm_analysis, llm_signal, current_price)
            else:
                # Log why no signal was sent
                logger.info(f"No alert sent. Signal: {llm_signal}, Confidence: {confidence}%")
                
        except Exception as e:
            logger.error(f"Error checking for signals: {e}")
            # Log full traceback for debugging
            import traceback
            traceback.print_exc()
    
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
    
    def run_continuous(self, interval_seconds=900):
        """
        Run the bot continuously with a specified check interval
        
        Args:
            interval_seconds: Interval between checks in seconds (default: 15 minutes)
        """
        # Format interval for display
        if interval_seconds < 60:
            display_interval = f"{interval_seconds} seconds"
        elif interval_seconds < 3600:
            minutes = interval_seconds / 60
            display_interval = f"{minutes:.1f} minute{'s' if minutes != 1 else ''}"
        else:
            hours = interval_seconds / 3600
            display_interval = f"{hours:.1f} hour{'s' if hours != 1 else ''}"
        
        console.print(Panel.fit(
            "[bold blue]Bitcoin Trading Signal Bot[/bold blue]\n\n"
            f"[bold]Symbol:[/bold] {self.symbol}\n"
            f"[bold]Timeframe:[/bold] {self.timeframe}\n"
            f"[bold]Check Interval:[/bold] Every {display_interval}\n",
            title="Starting Bot", border_style="green"
        ))
        
        # Run based on interval type
        if interval_seconds < 60:
            # For intervals less than a minute, use direct loop
            self.check_for_signals()  # Initial check
            
            while True:
                try:
                    time.sleep(interval_seconds)
                    self.check_for_signals()
                except KeyboardInterrupt:
                    logger.info("Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait a minute before trying again
        else:
            # For intervals of a minute or more, use schedule library
            minutes = interval_seconds / 60
            schedule.every(minutes).minutes.do(self.check_for_signals)
            
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
        
        # Add technical indicators based on configuration
        if self.use_advanced_indicators:
            try:
                from app.analysis.advanced_technical import AdvancedTechnicalAnalyzer
                df = AdvancedTechnicalAnalyzer.add_all_indicators(df)
                df = AdvancedTechnicalAnalyzer.detect_advanced_breakouts(df)
            except Exception as e:
                logger.error(f"Error applying advanced indicators in backtest, falling back to standard: {e}")
                df = TechnicalAnalyzer.add_indicators(df)
                df = TechnicalAnalyzer.detect_breakouts(df)
        else:
            df = TechnicalAnalyzer.add_indicators(df)
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
                # Determine which signal column to use based on available columns
                if self.use_advanced_indicators and 'advanced_long_signal' in df.columns:
                    long_signal_col = 'advanced_long_signal'
                    short_signal_col = 'advanced_short_signal'
                else:
                    long_signal_col = 'long_signal'
                    short_signal_col = 'short_signal'
                
                if row[long_signal_col] == 1:
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
                elif row[short_signal_col] == 1:
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