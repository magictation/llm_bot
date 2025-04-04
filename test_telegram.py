import os
import telebot
from rich.console import Console

# Create a console for rich output
console = Console()

def test_telegram_notification():
    """
    Comprehensive Telegram notification test script
    """
    # Get Telegram configuration
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

    # Detailed configuration check
    console.print("[bold yellow]Telegram Configuration Check:[/bold yellow]")
    console.print(f"Bot Token: {'*****' + (telegram_token[-5:] if telegram_token else 'Not Set')}")
    console.print(f"Chat ID: {telegram_chat_id or 'Not Set'}")

    # Validate configuration
    if not telegram_token:
        console.print("[red]Error: TELEGRAM_BOT_TOKEN is not set[/red]")
        return False

    if not telegram_chat_id:
        console.print("[red]Error: TELEGRAM_CHAT_ID is not set[/red]")
        return False

    try:
        # Initialize the bot
        bot = telebot.TeleBot(telegram_token)

        # Prepare test message
        test_message = """🤖 *Telegram Notification Test* 🤖

✅ Test Message Successful!
• Bot Token: Verified
• Chat ID: Verified
• Timestamp: {}

This is a test message to confirm Telegram bot functionality.""".format(
            os.getenv('TRADING_TIMEFRAME', 'Unknown')
        )

        # Send test message
        bot.send_message(
            chat_id=telegram_chat_id, 
            text=test_message,
            parse_mode='Markdown'
        )

        console.print("[green]✓ Test message sent successfully![/green]")
        return True

    except Exception as e:
        console.print("[red]Telegram Notification Test Failed:[/red]")
        console.print(f"[red]{str(e)}[/red]")
        return False

def main():
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Run the test
    test_telegram_notification()

if __name__ == "__main__":
    main()