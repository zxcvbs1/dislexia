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

def load_cached_data(symbol, cache_dir='data_cache'):
    """Load cached data if it exists"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}.csv")
    if os.path.exists(cache_file):
        print(f"Loading cached data for {symbol}")
        return pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
    return None

def save_cached_data(df, symbol, cache_dir='data_cache'):
    """Save data to cache"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}.csv")
    df.to_csv(cache_file)
    print(f"Cached data saved for {symbol}")

def fetch_historical_data(exchange, symbol, start_timestamp, end_timestamp, verbose=True):
    # First try to load from cache
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
    
    # Convert to DataFrame and cache
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Cache the data
    save_cached_data(df, symbol)
    
    return df

def find_nearest_timestamp(df, timestamp):
    """Find the nearest timestamp in the DataFrame"""
    timestamps = df.index.values
    idx = (np.abs(timestamps - np.datetime64(timestamp))).argmin()
    return df.index[idx]

def process_signals(signals_df, historical_data_dict):
    print("Processing trading signals...")
    results = []
    
    # Group signals by date and time
    signals_df['datetime'] = pd.to_datetime(signals_df['date'].astype(str) + ' ' + signals_df['time'])
    grouped_signals = signals_df.groupby(['date', 'time'])
    
    for (date, time), group in tqdm(grouped_signals, desc="Processing trading groups"):
        timestamp = pd.to_datetime(f"{date} {time}")
        
        # Initialize trade group
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
                # Find nearest timestamp using our custom function
                nearest_timestamp = find_nearest_timestamp(symbol_data, timestamp)
                entry_price = symbol_data.loc[nearest_timestamp, 'close']
                
                trades.append({
                    'symbol': symbol,
                    'position': row['position'],
                    'leverage': float(row['leverage']),
                    'entry_price': entry_price,
                    'entry_time': timestamp
                })
            except Exception as e:
                print(f"Warning: Error processing {symbol} at {timestamp}: {str(e)}")
                continue
        
        # Process trade group
        if trades:
            group_results = process_trade_group(trades, historical_data_dict, timestamp)
            results.extend(group_results)
    
    return pd.DataFrame(results)

def process_trade_group(trades, historical_data_dict, entry_time):
    results = []
    window_size = timedelta(hours=3)
    
    for trade in trades:
        symbol = trade['symbol']
        symbol_data = historical_data_dict[symbol]
        
        # Calculate exit conditions
        entry_price = trade['entry_price']
        stop_loss = entry_price * 0.5 if trade['position'] == 'LONG' else entry_price * 1.5
        take_profit = entry_price * 1.5 if trade['position'] == 'LONG' else entry_price * 0.5
        
        # Get data for 3-hour window
        window_end = entry_time + window_size
        window_data = symbol_data[symbol_data.index >= entry_time]
        window_data = window_data[window_data.index <= window_end]
        
        if window_data.empty:
            print(f"Warning: No price data found for {symbol} in the specified time window")
            continue
        
        # Initialize exit variables
        exit_price = window_data.iloc[-1]['close']
        exit_time = window_end
        exit_type = 'time_window'
        
        # Check for stop loss or take profit hits
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
        
        # Calculate PnL
        pnl_percentage = ((exit_price - entry_price) / entry_price * 100 * trade['leverage'])
        if trade['position'] == 'SHORT':
            pnl_percentage = -pnl_percentage
        
        results.append({
            'date': entry_time.date(),
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'position': trade['position'],
            'leverage': trade['leverage'],
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'exit_type': exit_type,
            'pnl_percentage': float(pnl_percentage)
        })
    
    return results

def main():
    print("Starting backtesting process...")
    
    print("\nReading signals.csv...")
    try:
        signals_df = pd.read_csv('backtest/transformed_signals.csv')
        signals_df['date'] = pd.to_datetime(signals_df['date'], format='%d/%m/%Y')
        print(f"Found {len(signals_df)} signals to process")
    except Exception as e:
        print(f"Error reading signals.csv: {str(e)}")
        return
    
    print("\nInitializing Binance connection...")
    exchange = ccxt.binance()
    
    symbols = signals_df['token'].unique()
    start_date = signals_df['date'].min()
    end_date = signals_df['date'].max()
    
    print(f"\nProcessing data from {start_date.date()} to {end_date.date()}")
    print(f"Total symbols to process: {len(symbols)}")
    
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    historical_data_dict = {}
    for symbol in symbols:
        if symbol == '1000SHIB':
            print(f"\nConverting 1000SHIB to SHIB...")
            symbol = 'SHIB'
        
        market_symbol = f"{symbol}/USDT"
        print(f"\nProcessing {market_symbol}")
        
        try:
            df = fetch_historical_data(exchange, market_symbol, start_timestamp, end_timestamp)
            historical_data_dict[symbol] = df
            print(f"Successfully processed {market_symbol}")
            
        except Exception as e:
            print(f"Error processing {market_symbol}: {str(e)}")
            continue
    
    print("\nProcessing trading signals...")
    results_df = process_signals(signals_df, historical_data_dict)
    
    print("\nSaving results to CSV...")
    try:
        results_df.to_csv('backtest_results.csv', index=False)
        print("Results saved successfully to 'backtest_results.csv'")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    print("\n=== Backtest Summary ===")
    print(f"Total Trades: {len(results_df)}")
    print(f"Profitable Trades: {len(results_df[results_df['pnl_percentage'] > 0])}")
    print(f"Average PnL: {results_df['pnl_percentage'].mean():.2f}%")
    win_rate = (len(results_df[results_df['pnl_percentage'] > 0]) / len(results_df) * 100)
    print(f"Win Rate: {win_rate:.2f}%")
    
    print("\n=== Results by Exit Type ===")
    exit_type_stats = results_df.groupby('exit_type').agg({
        'pnl_percentage': ['count', 'mean']
    })
    print(exit_type_stats)
    
    print("\nBacktesting completed!")

if __name__ == "__main__":
    main()