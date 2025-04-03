"""
Binance data fetcher for Bitcoin Trading Bot
"""
import datetime
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from app.config import logger

class BinanceDataFetcher:
    """Class to fetch data from Binance API"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize with optional API credentials"""
        # For data-only usage, we can use the client without authentication
        self.client = Client()
        logger.info("Initialized Binance client for public data access")
    
    def get_historical_klines(self, symbol: str, interval: str, 
                             start_time: str = None, 
                             end_time: str = None,
                             limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical klines/candlestick data and return as DataFrame
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '15m', '1d')
            start_time: Start time in 'YYYY-MM-DD' format
            end_time: End time in 'YYYY-MM-DD' format
            limit: Max number of records to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if start_time:
                start_ts = int(datetime.datetime.strptime(start_time, '%Y-%m-%d').timestamp() * 1000)
            else:
                start_ts = None
                
            if end_time:
                end_ts = int(datetime.datetime.strptime(end_time, '%Y-%m-%d').timestamp() * 1000)
            else:
                end_ts = None
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_ts,
                end_str=end_ts,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'taker_buy_base_asset_volume', 
                             'taker_buy_quote_asset_volume']
            
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} klines for {symbol} at {interval} interval")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch recent trades for a symbol"""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            df = pd.DataFrame(trades)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            raise
    
    def get_live_price(self, symbol: str) -> float:
        """Get current price of a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error fetching live price: {e}")
            raise