import pandas as pd
import ccxt
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import warnings
import os
import json
warnings.filterwarnings('ignore')

class TradingCalculator:
    def __init__(self, initial_equity, 
                 long_stop_loss_pct=2.5,
                 long_take_profit_pct=5.0,
                 short_stop_loss_pct=2.5,
                 short_take_profit_pct=5.0,
                 fixed_position_value=500):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.long_stop_loss_pct = long_stop_loss_pct / 100
        self.long_take_profit_pct = long_take_profit_pct / 100
        self.short_stop_loss_pct = short_stop_loss_pct / 100
        self.short_take_profit_pct = short_take_profit_pct / 100
        self.fixed_position_value = fixed_position_value

    def calculate_position_size(self, entry_price, leverage):
        """Calculate position size using fixed position value"""
        # Calculate margin requirement
        margin_requirement = self.fixed_position_value / leverage
        
        # Calculate position size in base currency
        position_size = self.fixed_position_value / entry_price
        
        return position_size, margin_requirement

    def calculate_pnl(self, trade_data):
        """Calculate PnL with improved calculations"""
        position_value = trade_data['position_size'] * trade_data['entry_price']
        exit_value = trade_data['position_size'] * trade_data['exit_price']
        
        # Calculate price change percentage
        price_change_pct = ((exit_value - position_value) / position_value) * 100
        
        # Calculate margin used
        margin_used = position_value / trade_data['leverage']
        
        # Calculate PnL amount
        pnl_amount = margin_used * (price_change_pct / 100) * trade_data['leverage']
        
        # For short positions, invert the PnL
        if trade_data['position'] == 'SHORT':
            pnl_amount = -pnl_amount
            price_change_pct = -price_change_pct
        
        # PnL percentage relative to margin
        pnl_percentage = price_change_pct * trade_data['leverage']
        
        return pnl_amount, pnl_percentage

def load_cached_data(symbol, cache_dir='data_cache'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}.csv")
    if os.path.exists(cache_file):
        print(f"Loading cached data for {symbol}")
        return pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
    return None

def save_cached_data(df, symbol, cache_dir='data_cache'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}.csv")
    df.to_csv(cache_file)
    print(f"Cached data saved for {symbol}")

def fetch_historical_data(exchange, symbol, start_timestamp, end_timestamp, verbose=True):
    cached_data = load_cached_data(symbol)
    if cached_data is not None:
        return cached_data
    
    timeframes = '1m'
    all_candles = []
    
    if verbose:
        print(f"\nFetching data for {symbol}...")
    
    current_timestamp = start_timestamp
    pbar = tqdm(total=(end_timestamp - start_timestamp) // 60000) if verbose else None
    
    while current_timestamp < end_timestamp:
        try:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframes,
                since=current_timestamp,
                limit=1000
            )
            
            if not candles:
                if verbose:
                    print(f"No more data available for {symbol}")
                break
                
            all_candles.extend(candles)
            current_timestamp = candles[-1][0] + 60000
            
            if verbose:
                pbar.update(len(candles))
            
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            break
    
    if verbose:
        pbar.close()
        print(f"Successfully downloaded {len(all_candles)} candles for {symbol}")
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    save_cached_data(df, symbol)
    return df

def find_nearest_timestamp(df, timestamp):
    timestamps = df.index.values
    idx = (np.abs(timestamps - np.datetime64(timestamp))).argmin()
    return df.index[idx]

def process_signals(signals_df, historical_data_dict, calculator):
    print("Processing trading signals...")
    results = []
    
    signals_df['datetime'] = pd.to_datetime(signals_df['date'].astype(str) + ' ' + signals_df['time'])
    grouped_signals = signals_df.groupby(['date', 'time'])
    
    for (date, time), group in tqdm(grouped_signals, desc="Processing trading groups"):
        timestamp = pd.to_datetime(f"{date} {time}")
        trades = []
        
        for _, row in group.iterrows():
            symbol = row['token']
            if symbol == '1000SHIB':
                symbol = 'SHIB'
            
            if symbol not in historical_data_dict:
                print(f"Warning: No historical data found for {symbol}")
                continue
                
            symbol_data = historical_data_dict[symbol]
            
            try:
                nearest_timestamp = find_nearest_timestamp(symbol_data, timestamp)
                entry_price = symbol_data.loc[nearest_timestamp, 'close']
                
                position_size, margin_requirement = calculator.calculate_position_size(
                    entry_price,
                    float(row['leverage'])
                )
                
                trades.append({
                    'symbol': symbol,
                    'position': row['position'],
                    'leverage': float(row['leverage']),
                    'entry_price': entry_price,
                    'entry_time': timestamp,
                    'position_size': position_size,
                    'margin_requirement': margin_requirement
                })
            except Exception as e:
                print(f"Warning: Error processing {symbol} at {timestamp}: {str(e)}")
                continue
        
        if trades:
            group_results = process_trade_group(trades, historical_data_dict, timestamp, calculator)
            results.extend(group_results)
    
    return pd.DataFrame(results)

def process_trade_group(trades, historical_data_dict, entry_time, calculator):
    results = []
    window_size = timedelta(hours=3)
    
    for trade in trades:
        symbol = trade['symbol']
        symbol_data = historical_data_dict[symbol]
        entry_price = trade['entry_price']
        
        # Calculate exit conditions based on position type
        if trade['position'] == 'LONG':
            stop_loss = entry_price * (1 - calculator.long_stop_loss_pct)
            take_profit = entry_price * (1 + calculator.long_take_profit_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + calculator.short_stop_loss_pct)
            take_profit = entry_price * (1 - calculator.short_take_profit_pct)
        
        window_end = entry_time + window_size
        window_data = symbol_data[symbol_data.index >= entry_time]
        window_data = window_data[window_data.index <= window_end]
        
        if window_data.empty:
            print(f"Warning: No price data found for {symbol} in the specified time window")
            continue
        
        exit_price = window_data.iloc[-1]['close']
        exit_time = window_end
        exit_type = 'time_window'
        
        for idx, row in window_data.iterrows():
            if trade['position'] == 'LONG':
                if row['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_time = idx
                    exit_type = 'stop_loss'
                    break
                elif row['high'] >= take_profit:
                    exit_price = take_profit
                    exit_time = idx
                    exit_type = 'take_profit'
                    break
            else:  # SHORT position
                if row['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_time = idx
                    exit_type = 'stop_loss'
                    break
                elif row['low'] <= take_profit:
                    exit_price = take_profit
                    exit_time = idx
                    exit_type = 'take_profit'
                    break
        
        trade_data = {
            **trade,
            'exit_price': exit_price,
            'exit_time': exit_time
        }
        
        final_pnl, pnl_percentage = calculator.calculate_pnl(trade_data)
        
        results.append({
            'date': entry_time.date(),
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'position': trade['position'],
            'leverage': trade['leverage'],
            'position_size': trade['position_size'],
            'margin_requirement': trade['margin_requirement'],
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'exit_type': exit_type,
            'pnl_amount': float(final_pnl),
            'pnl_percentage': float(pnl_percentage)
        })
    
    return results

def main():
    print("=== Backtesting System Starting ===")
    print(f"Start Time (UTC): {datetime.utcnow()}")
    
    # Initialize calculator with different SL/TP levels for long and short
    calculator = TradingCalculator(
        initial_equity=10000,
        long_stop_loss_pct=2.5,    # 2.5% stop loss for longs
        long_take_profit_pct=5.0,  # 5% take profit for longs
        short_stop_loss_pct=2.5,   # 2.5% stop loss for shorts
        short_take_profit_pct=5.0,  # 5% take profit for shorts
        fixed_position_value=500    # Fixed position size of $500
    )
    
    # Configuration parameters
    cache_dir = 'data_cache'
    results_dir = 'backtest_results'
    
    # Create necessary directories
    for directory in [cache_dir, results_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load signals file
    print("\nReading signals.csv...")
    try:
        signals_df = pd.read_csv('./transformed_signals.csv', sep=',')
        signals_df['date'] = pd.to_datetime(signals_df['date'], format='%d/%m/%Y')
        print(f"Found {len(signals_df)} signals to process")
        
        # Basic signal validation
        required_columns = ['date', 'time', 'token', 'position', 'leverage']
        missing_columns = [col for col in required_columns if col not in signals_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in signals.csv: {missing_columns}")
            
    except Exception as e:
        print(f"Error reading signals.csv: {str(e)}")
        return
    
    # Initialize exchange connection
    print("\nInitializing Binance connection...")
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Use futures market
            }
        })
    except Exception as e:
        print(f"Error initializing exchange: {str(e)}")
        return
    
    # Process date range and symbols
    symbols = signals_df['token'].unique()
    start_date = signals_df['date'].min()
    end_date = signals_df['date'].max()
    
    print("\n=== Backtest Parameters ===")
    print(f"Initial Equity: ${calculator.initial_equity:,.2f}")
    print(f"Fixed Position Size: ${calculator.fixed_position_value:,.2f}")
    print(f"Long Position - Stop Loss: {calculator.long_stop_loss_pct*100}%")
    print(f"Long Position - Take Profit: {calculator.long_take_profit_pct*100}%")
    print(f"Short Position - Stop Loss: {calculator.short_stop_loss_pct*100}%")
    print(f"Short Position - Take Profit: {calculator.short_take_profit_pct*100}%")
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print(f"Total Symbols: {len(symbols)}")
    
    # Convert dates to timestamps
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    # Fetch historical data for all symbols
    historical_data_dict = {}
    for symbol in symbols:
        # Handle special case for SHIB
        if symbol == '1000SHIB':
            print(f"\nConverting 1000SHIB to SHIB...")
            symbol = 'SHIB'
        
        market_symbol = f"{symbol}/USDT"
        print(f"\nProcessing {market_symbol}")
        
        try:
            df = fetch_historical_data(
                exchange=exchange,
                symbol=market_symbol,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                verbose=True
            )
            
            if df is not None and not df.empty:
                historical_data_dict[symbol] = df
                print(f"Successfully processed {market_symbol} - {len(df)} candles")
            else:
                print(f"Warning: No data retrieved for {market_symbol}")
                
        except Exception as e:
            print(f"Error processing {market_symbol}: {str(e)}")
            continue
    
    if not historical_data_dict:
        print("Error: No historical data could be retrieved. Exiting.")
        return
    
    # Process signals with new position sizing and PnL calculations
    print("\nProcessing trading signals...")
    try:
        results_df = process_signals(
            signals_df=signals_df,
            historical_data_dict=historical_data_dict,
            calculator=calculator
        )
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'backtest_results_{timestamp}.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during signal processing: {str(e)}")
        return
    
    # Calculate and display summary statistics
    print("\n=== Backtest Summary ===")
    print(f"Total Trades: {len(results_df)}")
    print(f"Initial Equity: ${calculator.initial_equity:,.2f}")
    
    final_equity = calculator.initial_equity + results_df['pnl_amount'].sum()
    print(f"Final Equity: ${final_equity:,.2f}")
    
    total_return = ((final_equity - calculator.initial_equity) / calculator.initial_equity) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    profitable_trades = len(results_df[results_df['pnl_amount'] > 0])
    print(f"Profitable Trades: {profitable_trades}")
    print(f"Loss-Making Trades: {len(results_df) - profitable_trades}")
    
    print(f"\nAverage PnL Amount: ${results_df['pnl_amount'].mean():.2f}")
    print(f"Average PnL Percentage: {results_df['pnl_percentage'].mean():.2f}%")
    
    win_rate = (profitable_trades / len(results_df)) * 100
    print(f"Win Rate: {win_rate:.2f}%")
    
    # Results by position type
    print("\n=== Results by Position Type ===")
    position_stats = results_df.groupby('position').agg({
        'pnl_amount': ['count', 'sum', 'mean'],
        'pnl_percentage': 'mean'
    }).round(2)
    print(position_stats)
    
    # Results by exit type
    print("\n=== Results by Exit Type ===")
    exit_type_stats = results_df.groupby('exit_type').agg({
        'pnl_amount': ['count', 'sum', 'mean'],
        'pnl_percentage': 'mean'
    }).round(2)
    print(exit_type_stats)
    
    # Calculate monthly returns
# Calculate monthly returns
    print("\n=== Monthly Returns ===")
    results_df['month'] = results_df['entry_time'].dt.to_period('M')
    monthly_returns = results_df.groupby('month')['pnl_amount'].sum()
    monthly_returns_pct = monthly_returns / calculator.initial_equity * 100
    print(monthly_returns_pct)
    
    # Convert monthly returns to a dictionary with string keys
    monthly_returns_dict = {str(k): float(v) for k, v in monthly_returns_pct.items()}

    # Calculate additional metrics
    max_drawdown = calculate_max_drawdown(results_df['pnl_amount'], calculator.initial_equity)
    sharpe_ratio = calculate_sharpe_ratio(results_df['pnl_percentage'])
    
    # Save summary statistics
    summary_data = {
        'backtest_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'initial_equity': calculator.initial_equity,
        'fixed_position_value': calculator.fixed_position_value,
        'long_stop_loss_pct': calculator.long_stop_loss_pct * 100,
        'long_take_profit_pct': calculator.long_take_profit_pct * 100,
        'short_stop_loss_pct': calculator.short_stop_loss_pct * 100,
        'short_take_profit_pct': calculator.short_take_profit_pct * 100,
        'final_equity': float(final_equity),
        'total_return': float(total_return),
        'total_trades': len(results_df),
        'profitable_trades': profitable_trades,
        'win_rate': float(win_rate),
        'avg_pnl_amount': float(results_df['pnl_amount'].mean()),
        'avg_pnl_percentage': float(results_df['pnl_percentage'].mean()),
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe_ratio),
        'monthly_returns': monthly_returns_dict
    }
    
    # Save summary to JSON
    summary_file = os.path.join(results_dir, f'backtest_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4, default=str)
    
    print(f"\nSummary statistics saved to: {summary_file}")
    print("\n=== Additional Metrics ===")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print("\nBacktesting completed!")

def calculate_max_drawdown(pnl_series, initial_equity):
    """Calculate the maximum drawdown from a series of PnL values"""
    cumulative = initial_equity + np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    return float(np.min(drawdown))

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=365*24):
    """Calculate the Sharpe Ratio for the strategy"""
    excess_returns = returns - risk_free_rate/periods_per_year
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std())

if __name__ == "__main__":
    main()