"""
Notification manager for Bitcoin Trading Bot
"""
from typing import Dict
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

class NotificationManager:
    """Class to handle notifications and alerts"""
    
    @staticmethod
    def console_alert(signal_type: str, price: float, analysis: Dict):
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