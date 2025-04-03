# Bitcoin Trading Signal Bot

![Bitcoin Trading](https://img.shields.io/badge/Bitcoin-Trading-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A powerful Bitcoin trading signal application that monitors price movements and generates SHORT/LONG intraday trading signals. Leveraging Binance's public data and AI-powered analysis, the bot identifies high-probability trading opportunities without requiring authenticated API access.

## 📊 Features

- **Market Data Monitoring**
  - Fetches Bitcoin price data from Binance's public API
  - Supports various timeframes (1m, 5m, 15m, 1h, 4h, etc.)
  - No API keys or authentication required

- **Technical Analysis Engine**
  - Calculates key technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
  - Detects breakouts based on volatility and price action
  - Identifies support and resistance levels dynamically

- **AI-Powered Analysis**
  - Integrates with Google's Gemini model for enhanced insights
  - Combines technical signals with AI-generated recommendations
  - Provides confidence scores for trading decisions

- **Smart Alerting System**
  - Generates notifications for high-confidence signals
  - Provides recommended stop-loss and take-profit levels
  - Includes detailed market analysis with each signal

- **Backtesting Framework**
  - Test strategies against historical data
  - Get detailed performance metrics
  - Visualize equity curves

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Internet connection (for Binance public data)
- Google Gemini API key (for AI analysis)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/bitcoin-trading-bot.git
cd bitcoin-trading-bot
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create a .env file**

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file:

```
# Gemini LLM API Configuration (required for AI-powered analysis)
GEMINI_API_KEY=your_gemini_api_key_here

# Trading Configuration
TRADING_SYMBOL=BTCUSDT
TRADING_TIMEFRAME=1h
LOOKBACK_DAYS=30
```

### Running the Bot

**Standard Mode**

```bash
python run.py run
```

**With Custom Parameters**

```bash
python run.py run --interval 30 --symbol BTCUSDT --timeframe 4h
```

**With Custom Environment File**

```bash
python run.py run --env .env.production
```

## 🧪 Backtesting

Test your strategy on historical data:

```bash
python run.py backtest --start 2023-01-01 --end 2023-12-31 --timeframe 1h
```

## 🐳 Docker Deployment

For containerized deployment:

1. **Build the Docker image**

```bash
docker build -t bitcoin-trading-bot .
```

2. **Run the container**

```bash
docker run -v $(pwd)/signals:/app/signals -v $(pwd)/.env:/app/.env bitcoin-trading-bot
```

3. **Using Docker Compose**

```bash
docker-compose up -d
```

## 🧩 Project Structure

```
bitcoin_trading_bot/
├── app/                     # Main application package
│   ├── __init__.py
│   ├── config.py            # Configuration handling
│   ├── data/                # Data fetching modules
│   │   ├── __init__.py
│   │   └── binance_client.py # Binance data fetcher (public API)
│   ├── analysis/            # Analysis modules
│   │   ├── __init__.py
│   │   ├── technical.py     # Technical analysis
│   │   └── llm.py           # LLM integration
│   ├── notification/        # Notification modules
│   │   ├── __init__.py
│   │   └── alerts.py        # Notification manager
│   └── bot/                 # Bot modules
│       ├── __init__.py
│       └── trading_bot.py   # Main bot class
├── logs/                    # Log files
├── signals/                 # Signal data storage
├── .env                     # Environment variables
├── .env.example             # Example environment file
├── requirements.txt         # Python dependencies
├── run.py                   # Main entry point
├── Dockerfile               # Docker configuration
└── docker-compose.yml       # Docker Compose config
```

## 🛠️ Customization

### Adjusting Technical Analysis Parameters

Edit `app/analysis/technical.py` to modify indicator calculations:

```python
# Example: Changing the RSI period from 14 to 21 days
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=21).mean()  # Changed from 14
loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean() # Changed from 14
```

### Modifying Breakout Detection

Customize breakout detection in the same file:

```python
# Example: Requiring higher volume for breakout confirmation
df['volume_surge'] = df['volume'] > df['volume_sma_20'] * 2.0  # Changed from 1.5
```

### Customizing LLM Prompts

Edit the prompts in `app/analysis/llm.py`:

```python
prompt = f"""
You are a professional cryptocurrency trader specializing in swing trading.
...
"""
```

## 📝 Usage Examples

### Basic Monitoring

```bash
python run.py run
```

This starts the bot with default settings:
- Symbol: BTCUSDT
- Timeframe: 1h
- Check interval: Every 15 minutes

### Day Trading Setup

```bash
python run.py run --timeframe 15m --interval 5
```

### Monitoring Ethereum

```bash
python run.py run --symbol ETHUSDT
```

### Long-term Analysis

```bash
python run.py run --timeframe 1d --interval 60
```

## 🚨 Troubleshooting

### Common Issues

**Binance Data Retrieval Problems**

```
Error fetching historical data: Connection error or rate limit exceeded
```

**Solution**: Binance may rate-limit public API requests. Try reducing the frequency of your requests or wait a few minutes before trying again.

**LLM API Issues**

```
Error during LLM analysis: API key not valid. Please pass a valid API key.
```

**Solution**: Verify your Google Gemini API key and make sure you have sufficient quota remaining.

### Logging

Check the logs directory for detailed error information:

```bash
cat logs/trading_bot.log
```

## ⚠️ Disclaimer

This software is for educational and informational purposes only. Use at your own risk. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. The developers are not responsible for any financial losses incurred from using this software.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Happy Trading! 🚀
