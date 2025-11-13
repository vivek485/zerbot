import streamlit as st
import requests
import pandas as pd
import time
import numpy as np

from datetime import datetime, timedelta, date
import ta
from ta.volatility import bollinger_hband,bollinger_lband,bollinger_mband
from ta.trend import sma_indicator
import logging
from typing import Dict, List, Optional, Tuple
import warnings
import threading

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class AdvancedTradingBot:
    def __init__(self, auth_token, start_date, end_date, symbols):
        # Configuration from Streamlit inputs
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = []
        self.symbol = symbols
        self.userid = 'ZF4455'
        self.timeframe = 'minute'
        self.auth_token = auth_token
        
        # Trading parameters
        self.max_orders = 2
        self.risk_per_trade = 1000  # Rs. 2000 per trade
        
        # Order tracking
        self.active_orders = []  # Track active order IDs
        self.current_trades = {
            'entry_price': None,
            'quantity': 0,
            'stop_loss': None,
            'target': None,
            'entry_time': None,
            'symbol': None,
            'buy_order_id': None,
            'sell_sl_order_id': None,
            'sell_target_order_id': None,
            'state': 'NO_TRADE'  # NO_TRADE, ENTRY_PENDING, ACTIVE_TRADE, EXIT_PENDING
        }
        
        # Bot control
        self.is_running = False
        self.thread = None
        
        # Initialize sound
        
        
        # Headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/144.0 Mobile/15E148 Safari/605.1.15',
            'Authorization': self.auth_token,
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        self.order_headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/144.0 Mobile/15E148 Safari/605.1.15',
            'X-Kite-Version': '3',
            'Authorization': self.auth_token,
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        # URLs
        self.order_url = 'https://kite.zerodha.com/oms/orders/regular'
        self.positions_url = 'https://kite.zerodha.com/oms/portfolio/positions'
        self.orders_url = 'https://kite.zerodha.com/oms/orders'
        
        self.initialize_symbols()
        logging.info("Trading Bot Initialized Successfully")

    def initialize_symbols(self):
        """Initialize instrument tokens for symbols"""
        for symbol in self.symbol:
            token = self.get_instrument_token(symbol)
            if token:
                logging.info(f"Instrument token for {symbol}: {token}")
                self.symbols.append(str(token))
            else:
                logging.error(f"Symbol '{symbol}' not found")

    def get_instrument_token(self, symbol):
        """Get instrument token for a trading symbol"""
        try:
            df = pd.read_csv('instruments.csv')
            result = df[df['tradingsymbol'] == symbol]
            
            if len(result) > 0:
                return result['instrument_token'].values[0]
            return None
        except Exception as e:
            logging.error(f"Error getting instrument token: {e}")
            return None



    def create_buy_order(self, symbol, quantity=1, order_type='MARKET', price=0, trigger_price=0):
        """Create a buy order dictionary"""
        return {
            'variety': 'regular',
            'exchange': 'NFO',
            'tradingsymbol': symbol,
            'transaction_type': 'BUY',
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'product': 'MIS',
            'validity': 'DAY',
            'disclosed_quantity': 0,
            'trigger_price': trigger_price,
            'squareoff': 0,
            'stoploss': 0,
            'trailing_stoploss': 0,
            'user_id': self.userid,
            'tag': 'trading_bot'
        }

    def create_sell_order(self, symbol, quantity=1, order_type='MARKET', price=0, trigger_price=0):
        """Create a sell order dictionary"""
        return {
            'variety': 'regular',
            'exchange': 'NFO',
            'tradingsymbol': symbol,
            'transaction_type': 'SELL',
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'product': 'MIS',
            'validity': 'DAY',
            'disclosed_quantity': 0,
            'trigger_price': trigger_price,
            'squareoff': 0,
            'stoploss': 0,
            'trailing_stoploss': 0,
            'user_id': self.userid,
            'tag': 'trading_bot'
        }

    def place_order(self, order_data):
        """Place an order and return order ID"""
        try:
            response = requests.post(self.order_url, headers=self.order_headers, data=order_data)
            if response.status_code == 200:
                order_response = response.json()
                if order_response['status'] == 'success':
                    order_id = order_response['data']['order_id']
                    self.active_orders.append(order_id)
                    logging.info(f"Order placed successfully: {order_id}")
                    return order_id
                else:
                    logging.error(f"Order failed: {order_response['message']}")
                    return None
            else:
                logging.error(f"HTTP error: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return None

    def cancel_order(self, order_id):
        """Cancel a specific order"""
        try:
            cancel_url = f'https://kite.zerodha.com/oms/orders/regular/{order_id}?order_id={order_id}&parent_order_id=&variety=regular'
            response = requests.delete(cancel_url, headers=self.order_headers)
            if response.status_code == 200:
                if order_id in self.active_orders:
                    self.active_orders.remove(order_id)
                logging.info(f"Order cancelled: {order_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error cancelling order: {e}")
            return False

    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order_id in self.active_orders[:]:
            self.cancel_order(order_id)

    def get_order_status(self, order_id):
        """Get status of a specific order"""
        try:
            response = requests.get(self.orders_url, headers=self.headers)
            if response.status_code == 200:
                orders_data = response.json()['data']
                for order in orders_data:
                    if str(order['order_id']) == str(order_id):
                        return order['status']
            return None
        except Exception as e:
            logging.error(f"Error getting order status: {e}")
            return None

    def get_positions(self):
        """Get current positions"""
        try:
            response = requests.get(self.positions_url, headers=self.headers)
            if response.status_code == 200:
                return response.json()['data']
            return None
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return None

    def calculate_heikin_ashi(self, df):
        """Calculate Heikin Ashi candles"""
        ha_df = df.copy()
        
        # Heikin Ashi Close
        ha_df['HA_Close'] = (ha_df['Open'] + ha_df['High'] + ha_df['Low'] + ha_df['Close']) / 4
        
        # Heikin Ashi Open
        ha_df['HA_Open'] = 0.0
        ha_open = (ha_df.iloc[0]['Open'] + ha_df.iloc[0]['Close']) / 2
        
        for i in range(len(ha_df)):
            if i == 0:
                ha_df.loc[i, 'HA_Open'] = ha_open
            else:
                ha_open = (ha_df.loc[i-1, 'HA_Open'] + ha_df.loc[i-1, 'HA_Close']) / 2
                ha_df.loc[i, 'HA_Open'] = ha_open
        
        # Heikin Ashi High and Low
        ha_df['HA_High'] = ha_df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        ha_df['HA_Low'] = ha_df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        return ha_df

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        if df.empty:
            return df
            
        df = self.calculate_heikin_ashi(df)
        
        # Replace with Heikin Ashi values
        df['Open'] = df['HA_Open']
        df['High'] = df['HA_High']
        df['Low'] = df['HA_Low']
        df['Close'] = df['HA_Close']
        
        # Bollinger Bands
        df['bb_middle'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Additional indicators for entry condition
        df['prev_low'] = df['Low'].shift(1)
        df['is_green'] = (df['Open'] < df['Close']).astype(int)
        df['is_red'] = (df['Open'] > df['Close']).astype(int)
        df['below_bb'] = ((df['prev_low'] < df['bb_lower']) & (df['High'] < df['bb_upper'])).astype(int) * 3
        df['entry_signal'] = df['below_bb'] + df['is_green'] + df['is_red'].shift(1)
        
        return df

    def safe_concat(self, df1, df2):
        """Safely concatenate two dataframes handling empty cases"""
        if df1.empty and df2.empty:
            return pd.DataFrame()
        elif df1.empty:
            return df2.copy()
        elif df2.empty:
            return df1.copy()
        else:
            # Ensure both dataframes have the same columns
            common_columns = list(set(df1.columns) & set(df2.columns))
            if common_columns:
                return pd.concat([df1[common_columns], df2[common_columns]], ignore_index=True)
            else:
                return df1.copy()

    def fetch_historical_data(self, token, symbol_name):
        """Fetch historical data for symbol"""
        columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'V', 'OI']
        final_df = pd.DataFrame(columns=columns)
        from_date = self.start_date

        while from_date < self.end_date:
            to_date = from_date + timedelta(days=30)
            url = f'https://kite.zerodha.com/oms/instruments/historical/{token}/{self.timeframe}?user_id={self.userid}&oi=1&from={from_date}&to={to_date}'
            
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    candles = data['data']['candles']
                    if candles:
                        temp_df = pd.DataFrame(candles, columns=columns)
                        # Ensure all expected columns are present
                        for col in columns:
                            if col not in temp_df.columns:
                                temp_df[col] = np.nan
                        final_df = self.safe_concat(final_df, temp_df)
            except Exception as e:
                logging.error(f"Error fetching data for {symbol_name}: {e}")
            
            from_date += timedelta(days=31)

        if final_df.empty:
            logging.warning(f"No data fetched for {symbol_name}")
            return pd.DataFrame()

        # Process timestamp
        try:
            final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce')
            final_df = final_df.dropna(subset=['timestamp'])
            
            if final_df.empty:
                return pd.DataFrame()
                
            final_df['date'] = final_df['timestamp'].dt.date
            final_df['time'] = final_df['timestamp'].dt.strftime('%H:%M')
            final_df['ticker'] = symbol_name
            
            # Select and reorder columns
            result_columns = ['ticker', 'date', 'time', 'Open', 'High', 'Low', 'Close', 'V', 'OI']
            available_columns = [col for col in result_columns if col in final_df.columns]
            
            return final_df[available_columns]
            
        except Exception as e:
            logging.error(f"Error processing timestamp for {symbol_name}: {e}")
            return pd.DataFrame()

    def check_entry_condition(self, df):
        """Check if entry condition is met"""
        if df.empty or len(df) < 2:
            return False, None
        
        # Ensure required columns exist
        required_columns = ['entry_signal', 'High', 'prev_low']
        if not all(col in df.columns for col in required_columns):
            return False, None
            
        last_candle = df.iloc[-2]
        #print(df)
        #df.to_csv(f'{self.symbol}_df.csv') # Previous completed candle
        return last_candle['entry_signal'] == 5, last_candle

    def calculate_position_size(self, stop_loss_price, entry_price):
        """Calculate position size based on risk"""
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            return 75  # Minimum quantity
        
        quantity = 75 #int(round((self.risk_per_trade / risk_per_unit) / 75)) * 75
        return max(quantity, 75)  # Minimum 75 shares

    def manage_entry(self, symbol, last_candle):
        """Manage entry logic"""
        if self.current_trades['state'] != 'NO_TRADE':
            return
        
        # Calculate entry parameters
        entry_high = last_candle['High']
        entry_price = entry_high + 1.0  # 1 point above high
        stop_loss = last_candle['prev_low'] - 0.5
        target = entry_price + (2 * (entry_price - stop_loss))
        
        quantity = self.calculate_position_size(stop_loss, entry_price)
        
        logging.info(f"Entry Signal Found - Symbol: {symbol}")
        logging.info(f"Entry Price: {entry_price}, SL: {stop_loss}, Target: {target}, Quantity: {quantity}")
        
        # Place limit order above high
        buy_order = self.create_buy_order(
            symbol=symbol,
            quantity=quantity,
            order_type='LIMIT',
            price=entry_price,
            trigger_price=0
        )
        
        order_id = self.place_order(buy_order)
        if order_id:
            self.current_trades.update({
                'symbol': symbol,
                'entry_price': entry_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'target': target,
                'buy_order_id': order_id,
                'state': 'ENTRY_PENDING'
            })
            self.play_sound()

    def check_pending_entry(self, current_high):
        """Check if pending entry needs to be converted to market order"""
        if (self.current_trades['state'] == 'ENTRY_PENDING' and 
            self.current_trades['entry_price'] is not None):
            
            # If price moved above our limit price, convert to market order
            if current_high > self.current_trades['entry_price']:
                logging.info("Price moved above limit price - converting to market order")
                
                # Cancel limit order
                if self.current_trades['buy_order_id']:
                    self.cancel_order(self.current_trades['buy_order_id'])
                
                # Place market order
                buy_order = self.create_buy_order(
                    symbol=self.current_trades['symbol'],
                    quantity=self.current_trades['quantity'],
                    order_type='MARKET',
                    price=0,
                    trigger_price=0
                )
                
                order_id = self.place_order(buy_order)
                if order_id:
                    self.current_trades['buy_order_id'] = order_id

    def manage_exit_orders(self):
        """Manage exit orders (SL and Target)"""
        if self.current_trades['state'] != 'ACTIVE_TRADE':
            return
        
        # Place initial SL order
        if not self.current_trades['sell_sl_order_id']:
            sl_order = self.create_sell_order(
                symbol=self.current_trades['symbol'],
                quantity=self.current_trades['quantity'],
                order_type='SL',
                price=self.current_trades['stop_loss'] - 0.5,
                trigger_price=self.current_trades['stop_loss']
            )
            
            order_id = self.place_order(sl_order)
            if order_id:
                self.current_trades['sell_sl_order_id'] = order_id
                logging.info(f"Initial SL order placed: {order_id}")

    def manage_trailing_stop(self, current_price):
        """Manage trailing stop logic"""
        if self.current_trades['state'] != 'ACTIVE_TRADE':
            return
        
        entry_price = self.current_trades['entry_price']
        target_price = self.current_trades['target']
        
        if entry_price is None or target_price is None:
            return
            
        fifty_percent_level = entry_price + (target_price - entry_price) * 0.5
        
        # Check if price crossed 50% of target
        if current_price >= fifty_percent_level:
            # Cancel existing SL order and place target order
            if (self.current_trades['sell_sl_order_id'] and 
                not self.current_trades['sell_target_order_id']):
                
                self.cancel_order(self.current_trades['sell_sl_order_id'])
                
                # Place target order
                target_order = self.create_sell_order(
                    symbol=self.current_trades['symbol'],
                    quantity=self.current_trades['quantity'],
                    order_type='LIMIT',
                    price=self.current_trades['target'],
                    trigger_price=0
                )
                
                order_id = self.place_order(target_order)
                if order_id:
                    self.current_trades['sell_target_order_id'] = order_id
                    self.current_trades['sell_sl_order_id'] = None
                    logging.info("Moved SL to target level")
        
        # Check if price fell back to entry price after crossing 50%
        elif (current_price <= entry_price and 
              self.current_trades['sell_target_order_id']):
            
            # Cancel target order and place SL order again
            self.cancel_order(self.current_trades['sell_target_order_id'])
            
            sl_order = self.create_sell_order(
                symbol=self.current_trades['symbol'],
                quantity=self.current_trades['quantity'],
                order_type='SL',
                price=self.current_trades['stop_loss'] - 0.5,
                trigger_price=self.current_trades['stop_loss']
            )
            
            order_id = self.place_order(sl_order)
            if order_id:
                self.current_trades['sell_sl_order_id'] = order_id
                self.current_trades['sell_target_order_id'] = None
                logging.info("Moved back to original SL")

    def update_trade_status(self):
        """Update trade status based on order status"""
        if self.current_trades['state'] == 'ENTRY_PENDING' and self.current_trades['buy_order_id']:
            status = self.get_order_status(self.current_trades['buy_order_id'])
            if status == 'COMPLETE':
                self.current_trades['state'] = 'ACTIVE_TRADE'
                self.current_trades['entry_time'] = datetime.now()
                logging.info("Buy order executed - Trade is now ACTIVE")
                self.manage_exit_orders()
            elif status in ['CANCELLED', 'REJECTED']:
                self.reset_trade()
        
        elif self.current_trades['state'] == 'ACTIVE_TRADE':
            # Check if any exit order is completed
            if self.current_trades['sell_sl_order_id']:
                sl_status = self.get_order_status(self.current_trades['sell_sl_order_id'])
                if sl_status == 'COMPLETE':
                    logging.info("SL order executed - Trade completed")
                    self.reset_trade()
            
            if self.current_trades['sell_target_order_id']:
                target_status = self.get_order_status(self.current_trades['sell_target_order_id'])
                if target_status == 'COMPLETE':
                    logging.info("Target order executed - Trade completed")
                    self.reset_trade()

    def reset_trade(self):
        """Reset trade to initial state"""
        # Cancel any pending orders
        if self.current_trades['buy_order_id']:
            self.cancel_order(self.current_trades['buy_order_id'])
        if self.current_trades['sell_sl_order_id']:
            self.cancel_order(self.current_trades['sell_sl_order_id'])
        if self.current_trades['sell_target_order_id']:
            self.cancel_order(self.current_trades['sell_target_order_id'])
            
        self.current_trades = {
            'entry_price': None,
            'quantity': 0,
            'stop_loss': None,
            'target': None,
            'entry_time': None,
            'symbol': None,
            'buy_order_id': None,
            'sell_sl_order_id': None,
            'sell_target_order_id': None,
            'state': 'NO_TRADE'
        }
        logging.info("Trade reset to NO_TRADE state")

    def wait_for_next_minute(self):
        """Wait until the next minute starts"""
        now = datetime.now()
        seconds_to_wait = 60 - now.second
        time.sleep(seconds_to_wait)

    def start_bot(self):
        """Start the trading bot in a separate thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_bot)
        self.thread.daemon = True
        self.thread.start()
        logging.info("Trading Bot Started")

    def stop_bot(self):
        """Stop the trading bot"""
        self.is_running = False
        self.cancel_all_orders()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logging.info("Trading Bot Stopped")

    def _run_bot(self):
        """Main trading loop (runs in thread)"""
        logging.info("Starting Trading Bot in thread...")
        
        while self.is_running:
            try:
                # Wait for next minute
                self.wait_for_next_minute()
                
                # Update trade status
                self.update_trade_status()
                
                # Process each symbol
                for i, (token, symbol_name) in enumerate(zip(self.symbols, self.symbol)):
                    if not self.is_running:
                        break
                        
                    if len(self.active_orders) >= self.max_orders:
                        logging.info("Maximum orders reached - skipping new entries")
                        continue
                    
                    # Fetch and process data
                    df = self.fetch_historical_data(token, symbol_name)
                    if df.empty:
                        logging.warning(f"No data available for {symbol_name}")
                        continue
                    
                    df = self.calculate_indicators(df)
                    if df.empty:
                        continue
                    
                    # Get current price from latest candle
                    current_candle = df.iloc[-1]
                    current_high = current_candle['High']
                    current_close = current_candle['Close']
                    
                    # Check entry condition
                    entry_condition, last_candle = self.check_entry_condition(df)
                    
                    if entry_condition and self.current_trades['state'] == 'NO_TRADE':
                        self.manage_entry(symbol_name, last_candle)
                    
                    # Check pending entry conversion
                    self.check_pending_entry(current_high)
                    
                    # Manage trailing stop for active trades
                    if self.current_trades['state'] == 'ACTIVE_TRADE':
                        self.manage_trailing_stop(current_close)
                
                # Log current status
                logging.info(f"Active Orders: {len(self.active_orders)}, Trade State: {self.current_trades['state']}")
                
                # Clean up completed orders from active_orders
                active_orders_copy = self.active_orders.copy()
                for order_id in active_orders_copy:
                    status = self.get_order_status(order_id)
                    if status in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                        self.active_orders.remove(order_id)
                
                time.sleep(1)  # Small delay to prevent excessive API calls
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(10)

    def get_bot_status(self):
        """Get current bot status for display"""
        return {
            'is_running': self.is_running,
            'active_orders': len(self.active_orders),
            'trade_state': self.current_trades['state'],
            'current_symbol': self.current_trades['symbol'],
            'entry_price': self.current_trades['entry_price'],
            'stop_loss': self.current_trades['stop_loss'],
            'target': self.current_trades['target']
        }


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Advanced Trading Bot",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Advanced Trading Bot")
    st.markdown("---")
    
    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Date inputs
        st.subheader("Date Range")
        from_date = st.date_input("From Date", value=date(2025, 11, 11))
        to_date = st.date_input("To Date", value=date(2025, 11, 10))
        
        # Enctoken input
        st.subheader("Authentication")
        auth_token = st.text_input(
            "Enctoken", 
            value="enctoken MZdXntzyore8z1/XsVVK9PZhvJXSsONCe2cnjeLhY5IEhsfbBR1Bs3yxLZ+mitbcjya0QxbouYPT+G9GCscD78qxUoeiDahEwPDukMOAoU1jcsuejZjO4w==",
            type="password"
        )
    
        # Symbol inputs
        st.subheader("Trading Symbols")
        symbol1 = st.text_input("Symbol 1", value="NIFTY25N1825950CE")
        symbol2 = st.text_input("Symbol 2", value="NIFTY25N1825900PE")
        
        symbols = [symbol1, symbol2]
        
        # Submit button
        if st.button("🚀 Initialize Trading Bot", type="primary", use_container_width=True):
            if from_date >= to_date:
                st.error("From Date must be before To Date")
            elif not auth_token:
                st.error("Please enter your enctoken")
            elif not symbol1 or not symbol2:
                st.error("Please enter both symbols")
            else:
                try:
                    # Convert dates to required format
                    start_date = from_date
                    end_date = to_date
                    
                    # Initialize bot
                    st.session_state.bot = AdvancedTradingBot(
                        auth_token=auth_token,
                        start_date=start_date,
                        end_date=end_date,
                        symbols=symbols
                    )
                    
                    st.success("✅ Trading Bot Initialized Successfully!")
                    st.info(f"Symbols: {symbols}")
                    st.info(f"Date Range: {start_date} to {end_date}")
                    
                except Exception as e:
                    st.error(f"Error initializing bot: {e}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Bot Control")
        
        if st.session_state.bot is not None:
            # Start/Stop buttons
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                if not st.session_state.bot_running:
                    if st.button("▶️ Start Bot", use_container_width=True):
                        st.session_state.bot.start_bot()
                        st.session_state.bot_running = True
                        st.rerun()
                else:
                    if st.button("⏹️ Stop Bot", use_container_width=True, type="secondary"):
                        st.session_state.bot.stop_bot()
                        st.session_state.bot_running = False
                        st.rerun()
            
            with col1_2:
                if st.button("🔄 Refresh Status", use_container_width=True):
                    st.rerun()
            
            # Display bot status
            st.subheader("Current Status")
            if st.session_state.bot_running:
                status = st.session_state.bot.get_bot_status()
                
                status_col1, status_col2, status_col3 = st.columns(3)
                
                with status_col1:
                    st.metric("Bot Status", "🟢 RUNNING")
                    st.metric("Active Orders", status['active_orders'])
                
                with status_col2:
                    st.metric("Trade State", status['trade_state'])
                    st.metric("Current Symbol", status['current_symbol'] or "None")
                
                with status_col3:
                    if status['entry_price']:
                        st.metric("Entry Price", f"₹{status['entry_price']:.2f}")
                        st.metric("Stop Loss", f"₹{status['stop_loss']:.2f}")
                        st.metric("Target", f"₹{status['target']:.2f}")
                    else:
                        st.metric("Entry Price", "N/A")
                        st.metric("Stop Loss", "N/A")
                        st.metric("Target", "N/A")
                
                # Live updates
                st.subheader("Live Updates")
                status_placeholder = st.empty()
                
                # Simulate live updates (you can replace this with actual live data)
                for i in range(5):
                    with status_placeholder.container():
                        st.info(f"Update {i+1}: Bot is monitoring symbols {symbols}")
                        time.sleep(1)
                        
            else:
                st.warning("⏸️ Bot is not running")
                st.info("Click 'Start Bot' to begin automated trading")
        
        else:
            st.info("👈 Please configure the bot in the sidebar and click 'Initialize Trading Bot'")
    
    with col2:
        st.header("Symbol Information")
        
        if st.session_state.bot is not None:
            st.success("✅ Bot Initialized")
            st.write("**Configured Symbols:**")
            for i, symbol in enumerate(symbols, 1):
                st.write(f"{i}. {symbol}")
            
            st.write("**Date Range:**")
            st.write(f"From: {from_date}")
            st.write(f"To: {to_date}")
            
            st.write("**Trading Parameters:**")
            st.write("- Max Orders: 2")
            st.write("- Risk per Trade: ₹2000")
            st.write("- Timeframe: 1 Minute")
            st.write("- Strategy: Heikin Ashi + Bollinger Bands")
        
        else:
            st.warning("Bot not initialized")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Advanced Trading Bot | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
