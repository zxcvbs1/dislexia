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
    def __init__(self, initial_equity, allocation_per_timeframe=0.10):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.allocation_per_timeframe = allocation_per_timeframe
        
    def calculate_position_size(self, entry_price, leverage, num_simultaneous_trades):
        """
        Calculate position size based on:
        - Available equity per timeframe
        - Number of simultaneous trades
        - Leverage and margin requirements
        """
        # Available equity for this timeframe
        available_equity = self.current_equity * self.allocation_per_timeframe
        
        # Divide among simultaneous trades
        equity_per_trade = available_equity / num_simultaneous_trades
        
        # Calculate margin requirement (assuming 1/leverage as margin requirement)
        margin_requirement = 1 / leverage
        
        # Calculate maximum position size considering leverage and margin
        max_position_size = (equity_per_trade * leverage) / entry_price
        
        return max_position_size, equity_per_trade

    def calculate_funding_rate_cost(self, position_size, entry_price, funding_rate, hold_time_hours):
        """Calculate the cost of funding rates"""
        # Funding rates are typically charged every 8 hours
        funding_intervals = hold_time_hours / 8
        position_value = position_size * entry_price
        funding_cost = position_value * funding_rate * funding_intervals
        return funding_cost

    def calculate_pnl(self, trade_data):
        """
        Calculate PnL considering:
        - Position size
        - Leverage
        - Funding rates
        - Margin requirements
        """
        position_value = trade_data['position_size'] * trade_data['entry_price']
        exit_value = trade_data['position_size'] * trade_data['exit_price']
        
        # Calculate raw PnL
        raw_pnl = exit_value - position_value
        
        # Calculate holding period in hours
        hold_time = (trade_data['exit_time'] - trade_data['entry_time']).total_seconds() / 3600
        
        # Calculate funding rate cost (assuming 0.01% per 8 hours as example)
        funding_cost = self.calculate_funding_rate_cost(
            trade_data['position_size'],
            trade_data['entry_price'],
            0.0001,  # 0.01% funding rate
            hold_time
        )
        
        # Calculate final PnL
        final_pnl = raw_pnl - funding_cost
        
        # Calculate PnL percentage relative to margin used
        margin_used = position_value / trade_data['leverage']
        pnl_percentage = (final_pnl / margin_used) * 100
        
        if trade_data['position'] == 'SHORT':
            pnl_percentage = -pnl_percentage
            
        return final_pnl, pnl_percentage


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

def process_signals(signals_df, historical_data_dict, initial_equity=10000):
    print("Processing trading signals...")
    results = []
    
    # Initialize trading calculator
    calculator = TradingCalculator(initial_equity)
    
    # Group signals by date and time
    signals_df['datetime'] = pd.to_datetime(signals_df['date'].astype(str) + ' ' + signals_df['time'])
    grouped_signals = signals_df.groupby(['date', 'time'])
    
    for (date, time), group in tqdm(grouped_signals, desc="Processing trading groups"):
        timestamp = pd.to_datetime(f"{date} {time}")
        
        # Initialize trade group
        trades = []
        num_trades = len(group)
        
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
                
                # Calculate position size for this trade
                position_size, equity_per_trade = calculator.calculate_position_size(
                    entry_price,
                    float(row['leverage']),
                    num_trades
                )
                
                trades.append({
                    'symbol': symbol,
                    'position': row['position'],
                    'leverage': float(row['leverage']),
                    'entry_price': entry_price,
                    'entry_time': timestamp,
                    'position_size': position_size,
                    'equity_allocated': equity_per_trade
                })
            except Exception as e:
                print(f"Warning: Error processing {symbol} at {timestamp}: {str(e)}")
                continue
        
        # Process trade group
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
        
        # Calculate exit conditions
        entry_price = trade['entry_price']
        stop_loss = entry_price * 0.5 if trade['position'] == 'LONG' else entry_price * 1.5
        take_profit = entry_price * 1.5 if trade['position'] == 'LONG' else entry_price * 0.5
        
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
        
        # Calculate PnL with all factors considered
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
            'equity_allocated': trade['equity_allocated'],
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
    
    # Configuration parameters
    initial_equity = 10000  # Starting with $10,000
    cache_dir = 'data_cache'
    results_dir = 'backtest_results'
    
    # Create necessary directories
    for directory in [cache_dir, results_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load signals file
    print("\nReading signals.csv...")
    try:
        signals_df = pd.read_csv('backtest/transformed_signals.csv', sep=',')
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
    print(f"Initial Equity: ${initial_equity:,.2f}")
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print(f"Total Symbols: {len(symbols)}")
    print(f"Allocation per timeframe: 10% (${initial_equity * 0.10:,.2f})")
    
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
            initial_equity=initial_equity
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
    print(f"Initial Equity: ${initial_equity:,.2f}")
    
    final_equity = initial_equity + results_df['pnl_amount'].sum()
    print(f"Final Equity: ${final_equity:,.2f}")
    
    total_return = ((final_equity - initial_equity) / initial_equity) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    profitable_trades = len(results_df[results_df['pnl_amount'] > 0])
    print(f"Profitable Trades: {profitable_trades}")
    print(f"Loss-Making Trades: {len(results_df) - profitable_trades}")
    
    print(f"\nAverage PnL Amount: ${results_df['pnl_amount'].mean():.2f}")
    print(f"Average PnL Percentage: {results_df['pnl_percentage'].mean():.2f}%")
    
    win_rate = (profitable_trades / len(results_df)) * 100
    print(f"Win Rate: {win_rate:.2f}%")
    
    # Calculate and display results by exit type
    print("\n=== Results by Exit Type ===")
    exit_type_stats = results_df.groupby('exit_type').agg({
        'pnl_amount': ['count', 'sum', 'mean'],
        'pnl_percentage': 'mean'
    }).round(2)
    print(exit_type_stats)
    
# Calculate monthly returns
    print("\n=== Monthly Returns ===")
    results_df['month'] = results_df['entry_time'].dt.to_period('M')
    monthly_returns = results_df.groupby('month')['pnl_amount'].sum()
    monthly_returns_pct = monthly_returns / initial_equity * 100
    print(monthly_returns_pct)
    
    # Convert monthly returns to a dictionary with string keys
    monthly_returns_dict = {str(k): float(v) for k, v in monthly_returns_pct.items()}

    # Save summary statistics
    summary_data = {
        'backtest_date': datetime.utcnow(),
        'initial_equity': initial_equity,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_trades': len(results_df),
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'avg_pnl_amount': float(results_df['pnl_amount'].mean()),
        'avg_pnl_percentage': float(results_df['pnl_percentage'].mean()),
        'max_drawdown': calculate_max_drawdown(results_df['pnl_amount'], initial_equity),
        'sharpe_ratio': calculate_sharpe_ratio(results_df['pnl_percentage']),
        'monthly_returns': monthly_returns_dict  # Use the converted dictionary
    }
    
    summary_file = os.path.join(results_dir, f'backtest_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4, default=str)


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