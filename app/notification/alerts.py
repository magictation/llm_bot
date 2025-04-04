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
        Send alert to Telegram
        
        Args:
            signal_type: Type of signal (LONG/SHORT)
            price: Current price
            analysis: LLM analysis dictionary
        """
        # Detailed logging for troubleshooting
        console.print("[bold yellow]Telegram Alert Attempt:[/bold yellow]")
        console.print(f"Telegram Bot: {'Initialized' if self.telegram_bot else 'Not Initialized'}")
        console.print(f"Chat ID: {self.telegram_chat_id or 'Not Set'}")
        
        # Log detailed information about the alert attempt
        console.print("[bold yellow]Telegram Alert Details:[/bold yellow]")
        console.print(f"Signal Type: {signal_type}")
        console.print(f"Price: ${price:.2f}")
        console.print(f"Confidence: {analysis.get('confidence', 'N/A')}%")
        console.print(f"Market Condition: {analysis.get('market_condition', 'Unknown')}")

        # Check if Telegram bot is properly configured
        if not self.telegram_bot or not self.telegram_chat_id:
            console.print("[red]Telegram bot not fully configured. Skipping Telegram alert.[/red]")
            return
        
        # Determine emoji and title based on signal type
        if signal_type == "LONG":
            signal_emoji = "🟢 LONG 📈"
        elif signal_type == "SHORT":
            signal_emoji = "🔴 SHORT 📉"
        else:
            signal_emoji = "⚪ NEUTRAL ➡️"
        
        # Truncate reasoning if it's too long
        reasoning = analysis.get('reasoning', 'No detailed analysis available')
        if len(reasoning) > 300:
            reasoning = reasoning[:300] + '...'
        
        # Construct message
        message = f"""🤖 Bitcoin Trading Signal 🤖

                        {signal_emoji}
                        Confidence: {analysis.get('confidence', 'N/A')}%
                        Current Price: ${price:.2f}

                        Market Condition: {analysis.get('market_condition', 'Unknown')}

                        🛡️ Risk Management:
                        Stop Loss: ${analysis.get('stop_loss', 'N/A'):.2f}

                        🎯 Take Profit Targets:
                        {''.join([f"✅ TP{i+1}: ${tp:.2f}\n" for i, tp in enumerate(analysis.get('take_profit', []))])}
                        Risk/Reward: {analysis.get('risk_reward_ratio', 'N/A')}

                        📝 Analysis:
                        {reasoning}
                        """
        try:
            # Send message to Telegram
            self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id, 
                text=message,
                parse_mode='Markdown'
            )
            console.print("[green]Telegram alert sent successfully[/green]")
        except Exception as e:
            console.print(f"[red]Failed to send Telegram alert: {e}[/red]")
            # Log the full error for debugging
            import traceback
            traceback.print_exc()