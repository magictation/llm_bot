"""
Notification manager for Bitcoin Trading Signal Bot
"""
import os
import telebot
from typing import Dict
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

class NotificationManager:
    """Class to handle notifications and alerts"""
    
    def __init__(self):
        """Initialize Telegram bot if token is available"""
        # Get Telegram configuration from environment variables
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Initialize Telegram-related attributes
        self.telegram_bot = None
        self.telegram_chat_id = None
        
        # Detailed logging for Telegram configuration
        console.print("[bold yellow]Telegram Configuration Check:[/bold yellow]")
        console.print(f"Bot Token: {'*****' + (telegram_token[-5:] if telegram_token else 'Not Set')}")
        console.print(f"Chat ID: {telegram_chat_id or 'Not Set'}")
        
        # Initialize Telegram bot if both token and chat ID are present
        if telegram_token and telegram_chat_id:
            try:
                self.telegram_bot = telebot.TeleBot(telegram_token)
                self.telegram_chat_id = telegram_chat_id
                
                # Verify bot initialization
                console.print("[green]Telegram bot initialized successfully[/green]")
            except Exception as e:
                console.print(f"[red]Failed to initialize Telegram bot: {e}[/red]")
                # Log the full error for debugging
                import traceback
                traceback.print_exc()
        else:
            console.print("[red]Telegram bot not configured. Missing token or chat ID.[/red]")
    
    def console_alert(self, signal_type: str, price: float, analysis: Dict):
        """
        Display alert in console
        
        Args:
            signal_type: Type of signal (LONG/SHORT)
            price: Current price
            analysis: LLM analysis dictionary
        """
        if signal_type == "LONG":
            color = "green"
            title = "🚀 LONG SIGNAL DETECTED"
        elif signal_type == "SHORT":
            color = "red"
            title = "🔻 SHORT SIGNAL DETECTED"
        else:
            color = "yellow"
            title = "⚠️ SIGNAL UPDATE"
        
        panel_text = Text()
        panel_text.append(f"Current BTC Price: ${price:.2f}\n\n", style="bold")
        panel_text.append(f"Signal: ", style="bold")
        panel_text.append(f"{analysis['signal']} (Confidence: {analysis['confidence']}%)\n", style=color)
        panel_text.append(f"Market Condition: ", style="bold")
        panel_text.append(f"{analysis['market_condition']}\n\n", style="italic")
        
        panel_text.append("Risk Management:\n", style="bold underline")
        panel_text.append(f"Stop Loss: ${analysis['stop_loss']:.2f}\n", style="bold red")
        
        panel_text.append("Take Profit Targets:\n", style="bold underline")
        for i, tp in enumerate(analysis['take_profit']):
            panel_text.append(f"TP{i+1}: ${tp:.2f}\n", style="bold green")
        
        panel_text.append(f"Risk/Reward: {analysis['risk_reward_ratio']:.2f}\n\n", style="bold")
        
        panel_text.append("Analysis:\n", style="bold underline")
        panel_text.append(f"{analysis['reasoning']}", style="italic")
        
        console.print(Panel(panel_text, title=title, border_style=color, expand=False))
    
    def telegram_alert(self, signal_type: str, price: float, analysis: Dict):
        """
        Send alert to Telegram with enhanced market analysis
        
        Args:
            signal_type: Type of signal (LONG/SHORT)
            price: Current price
            analysis: LLM analysis dictionary (enhanced version with multi-step analysis)
        """
        # Check if Telegram bot is properly configured
        if not self.telegram_bot or not self.telegram_chat_id:
            console.print("[red]Telegram bot not fully configured. Skipping Telegram alert.[/red]")
            return
        
        try:
            # Determine emoji based on signal type
            if signal_type == "LONG":
                signal_emoji = "🟢 LONG 📈"
            elif signal_type == "SHORT":
                signal_emoji = "🔴 SHORT 📉"
            else:
                signal_emoji = "⚪ NEUTRAL ➡️"
            
            # Get analysis text - don't truncate it
            reasoning = analysis.get('reasoning', 'No detailed analysis available')
            
            # Check for reversal information in the reasoning text
            is_reversal = "REVERSAL ALERT" in reasoning or "reversal" in reasoning.lower()
            reversal_emoji = "🔄" if is_reversal else ""
            
            # Format take profit targets compactly
            take_profit_text = "None"
            take_profits = analysis.get('take_profit', [])
            if take_profits:
                take_profit_text = ", ".join([f"${tp:.2f}" for tp in take_profits[:5]])
                if len(take_profits) > 5:
                    take_profit_text += f"... ({len(take_profits)-5} more)"
            
            # Format resistance levels
            resistance_levels = analysis.get('resistance_levels', [])
            resistance_text = "None detected"
            if resistance_levels and len(resistance_levels) > 0:
                # Sort resistance levels from highest to lowest
                sorted_resistance = sorted(resistance_levels, reverse=True)
                resistance_text = ", ".join([f"${level:.2f}" for level in sorted_resistance[:5]])
                if len(sorted_resistance) > 5:
                    resistance_text += f"... ({len(sorted_resistance)-5} more)"
            
            # Format support levels
            support_levels = analysis.get('support_levels', [])
            support_text = "None detected"
            if support_levels and len(support_levels) > 0:
                # Sort support levels from lowest to highest
                sorted_support = sorted(support_levels)
                support_text = ", ".join([f"${level:.2f}" for level in sorted_support[:5]])
                if len(sorted_support) > 5:
                    support_text += f"... ({len(sorted_support)-5} more)"
            
            # Get market assessment information from enhanced analysis
            market_assessment = analysis.get('market_assessment', {})
            technical_analysis = analysis.get('technical_analysis', {})
            
            # Add reversal section if detected
            reversal_section = ""
            if is_reversal:
                reversal_section = f"⚠️ REVERSAL ALERT! ⚠️\nPrevious trend may be reversing\n\n"
            
            # Calculate distance to key levels
            closest_support = min(support_levels, key=lambda x: abs(price - x)) if support_levels else None
            closest_resistance = min(resistance_levels, key=lambda x: abs(price - x)) if resistance_levels else None
            
            support_distance = (price - closest_support) / price * 100 if closest_support else None
            resistance_distance = (closest_resistance - price) / price * 100 if closest_resistance else None
            
            # Create the main signal message
            main_message = (
                f"🤖 Bitcoin Trading Signal {reversal_emoji} 🤖\n\n"
                f"{reversal_section}"
                f"{signal_emoji}\n"
                f"Confidence: {analysis.get('confidence', 'N/A')}%\n"
                f"Current Price: ${price:.2f}\n\n"
            )
            
            # Add market phase if available
            if 'market_phase' in market_assessment:
                main_message += f"Market Phase: {market_assessment['market_phase'].title()}\n"
                
            # Add trend strength if available
            if 'trend_strength' in market_assessment:
                strength = market_assessment['trend_strength']
                strength_emoji = "💪" if strength > 70 else "👍" if strength > 40 else "👋"
                main_message += f"Trend Strength: {strength_emoji} {strength}%\n"
                
            # Add market condition from main analysis
            main_message += f"Market Condition: {analysis.get('market_condition', 'Unknown')}\n\n"
            
            # Key levels section
            main_message += f"🔑 Key Levels:\n"
            main_message += f"Resistance: {resistance_text}\n"
            main_message += f"Support: {support_text}\n\n"
            
            # Add level distance information
            if closest_support and closest_resistance:
                main_message += f"📏 Price Distance:\n"
                main_message += f"→ {abs(support_distance):.2f}% to support (${closest_support:.2f})\n"
                main_message += f"→ {abs(resistance_distance):.2f}% to resistance (${closest_resistance:.2f})\n\n"
            
            # Risk management section
            main_message += f"📊 Risk Management:\n"
            main_message += f"Stop Loss: ${analysis.get('stop_loss', 'N/A'):.2f}\n"
            main_message += f"Take Profit: {take_profit_text}\n"
            main_message += f"Risk/Reward: {analysis.get('risk_reward_ratio', 'N/A')}\n\n"
            
            # Technical signals section
            if technical_analysis and 'technical_signals' in technical_analysis:
                signals = technical_analysis['technical_signals']
                main_message += f"📈 Technical Signals:\n"
                
                if 'rsi_signal' in signals:
                    rsi_emoji = "🔥" if signals['rsi_signal'] == "overbought" else "❄️" if signals['rsi_signal'] == "oversold" else "⚖️"
                    main_message += f"RSI: {rsi_emoji} {signals['rsi_signal']}\n"
                    
                if 'macd_signal' in signals:
                    macd_emoji = "📈" if signals['macd_signal'] == "bullish" else "📉" if signals['macd_signal'] == "bearish" else "↔️"
                    main_message += f"MACD: {macd_emoji} {signals['macd_signal']}\n"
                    
                if 'bollinger_signal' in signals:
                    bb_emoji = "🔝" if "upper" in signals['bollinger_signal'] else "🔙" if "lower" in signals['bollinger_signal'] else "↔️"
                    main_message += f"Bollinger: {bb_emoji} {signals['bollinger_signal']}\n"
            
            # Warning signs if available
            if 'warning_signs' in analysis and analysis['warning_signs']:
                warnings = analysis['warning_signs']
                if warnings and len(warnings) > 0:
                    main_message += f"\n⚠️ Watch Out For:\n"
                    for warning in warnings[:3]:
                        main_message += f"- {warning}\n"
            
            # Send the main signal message
            self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id, 
                text=main_message
            )
            
            # Create and send analysis as a separate message
            analysis_message = f"📝 Analysis:\n{reasoning}"
            
            # Split analysis into chunks if needed (Telegram has 4096 char limit)
            max_length = 4000  # Leave some margin for safety
            
            if len(analysis_message) <= max_length:
                # Send as a single message if it fits
                self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id, 
                    text=analysis_message
                )
            else:
                # Split into multiple messages if too long
                chunks = []
                current_chunk = "📝 Analysis (part 1):\n"
                
                # Split by sentences to avoid cutting mid-sentence
                sentences = reasoning.replace('. ', '.|').split('|')
                
                part_num = 1
                for sentence in sentences:
                    # If adding this sentence would make the chunk too long, start a new chunk
                    if len(current_chunk) + len(sentence) + 5 > max_length:
                        chunks.append(current_chunk)
                        part_num += 1
                        current_chunk = f"📝 Analysis (part {part_num}):\n"
                    
                    current_chunk += sentence + ". "
                
                # Add the last chunk if it has content
                if current_chunk and current_chunk != f"📝 Analysis (part {part_num}):\n":
                    chunks.append(current_chunk)
                
                # Send each chunk as a separate message
                for chunk in chunks:
                    self.telegram_bot.send_message(
                        chat_id=self.telegram_chat_id, 
                        text=chunk
                    )
                    
                    # Small delay to maintain message order
                    import time
                    time.sleep(0.5)
            
            # Send probable scenarios from technical analysis as a third message
            if technical_analysis and 'probable_scenarios' in technical_analysis:
                scenarios = technical_analysis['probable_scenarios']
                if scenarios and len(scenarios) > 0:
                    scenarios_message = "🔮 Probable Scenarios:\n\n"
                    
                    for i, scenario in enumerate(scenarios, 1):
                        # Handle both string scenarios and dictionary/object scenarios
                        if isinstance(scenario, dict):
                            # Extract key information
                            title = scenario.get('scenario', '')
                            description = scenario.get('description', '')
                            price_target = scenario.get('price_target', '')
                            probability = ''
                            
                            # Extract probability if it's in the title
                            if '%' in title:
                                probability_start = title.find('(')
                                if probability_start > 0:
                                    probability = title[probability_start:].strip()
                                    title = title[:probability_start].strip()
                            
                            # Format in a readable way
                            scenarios_message += f"{i}. {title}"
                            if probability:
                                scenarios_message += f" {probability}"
                            scenarios_message += "\n"
                            
                            if price_target:
                                scenarios_message += f"Target: {price_target}\n"
                            
                            if description:
                                # Format the description text
                                desc_lines = []
                                current_line = ""
                                
                                # Break description into readable lines
                                for word in description.split():
                                    if len(current_line + " " + word) > 60:  # Reasonable line length
                                        desc_lines.append(current_line)
                                        current_line = word
                                    else:
                                        current_line += " " + word if current_line else word
                                        
                                if current_line:
                                    desc_lines.append(current_line)
                                    
                                # Add description with proper indentation
                                for line in desc_lines:
                                    scenarios_message += f"   {line}\n"
                        else:
                            # Handle string scenarios
                            scenarios_message += f"{i}. {scenario}\n"
                        
                        # Add spacing between scenarios
                        scenarios_message += "\n"
                    
                    # Send the scenarios message
                    self.telegram_bot.send_message(
                        chat_id=self.telegram_chat_id, 
                        text=scenarios_message
                    )
            
            console.print("[green]Telegram alert sent successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Failed to send Telegram alert: {e}[/red]")
            import traceback
            traceback.print_exc()
            
            try:
                # Send a minimal emergency message if everything else fails
                emergency_message = (
                    f"SIGNAL: {signal_type} @ ${price:.2f}\n"
                    f"SL: ${analysis.get('stop_loss', 'N/A'):.2f}"
                )
                self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id, 
                    text=emergency_message
                )
            except:
                pass  # If this fails too, we've done all we can