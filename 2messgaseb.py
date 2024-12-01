import re
import json
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(
    filename='signal_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Predefined set of known tokens and their variations
KNOWN_TOKENS = {
    'BTC': ['BTC', 'BITCOIN', 'BTCUSDT', 'BITCOIN BTCUSDT', 'BITCOIN(BTC)', 'BTC/USDT'],
    'ETH': ['ETH', 'ETHEREUM', 'ETHUSDT', 'ETHEREUM ETHUSDT', 'ETHEREUM(ETH)', 'ETH/USDT'],
    'XRP': ['XRP', 'RIPPLE', 'XRPUSDT', 'RIPPLE XRPUSDT', 'RIPPLE(XRP)', 'XRP/USDT'],
    'BNB': ['BNB', 'BINANCE', 'BNBUSDT', 'BINANCE COIN', 'COIN', 'BNB/USDT'],
    'ADA': ['ADA', 'CARDANO', 'ADAUSDT', 'CARDANO ADAUSDT', 'CARDANO(ADA)', 'ADA/USDT'],
    'SOL': ['SOL', 'SOLANA', 'SOLUSDT', 'SOLANA SOLUSDT', 'SOLANA(SOL)', 'SOL/USDT'],
    'DOGE': ['DOGE', 'DOGECOIN', 'DOGEUSDT', 'DOGECOIN DOGEUSDT', 'DOGECOIN(DOGE)', 'DOGE/USDT'],
    'LTC': ['LTC', 'LITECOIN', 'LTCUSDT', 'LITECOIN LTCUSDT', 'LITECOIN(LTC)', 'LTC/USDT'],
    'TON': ['TON', 'TONCOIN', 'TONUSDT', 'TONCOIN TONUSDT', 'TONCOIN(TON)', 'TON/USDT'],
    'FTM': ['FTM', 'FANTOM', 'FTMUSDT', 'FANTOM FTMUSDT', 'FANTOM(FTM)', 'FTM/USDT'],
    'SUI': ['SUI', 'SUIUSDT', 'SUI/USDT'],
    'TRX': ['TRX', 'TRON', 'TRXUSDT', 'TRON TRXUSDT', 'TRON(TRX)', 'TRX/USDT'],
    'ICP': ['ICP', 'ICPUSDT', 'ICP/USDT'],
    'STX': ['STX', 'STXUSDT', 'STX/USDT'],
    'GRT': ['GRT', 'GRTUSDT', 'GRT/USDT'],
    'AVAX': ['AVAX', 'AVAXUSDT', 'AVAX/USDT'],
    'UNI': ['UNI', 'UNIUSDT', 'UNI/USDT'],
    'TAO': ['TAO', 'TAOUSDT', 'TAO/USDT'],
    '1000SHIB': ['1000SHIB', '1000SHIBUSDT', 'SHIB', 'SHIBUSDT', 'SHIB/USDT']
}

# Create reverse lookup for token normalization
TOKEN_LOOKUP = {variant.upper(): standard 
                for standard, variants in KNOWN_TOKENS.items() 
                for variant in variants}

class SignalProcessingError:
    def __init__(self):
        self.total_messages = 0
        self.successful_messages = 0
        self.error_messages = 0
        self.token_mismatch_errors = []
        self.parsing_errors = []
        self.current_user = os.getenv('USER', 'zxcvbs1')
        self.current_time = datetime.utcnow()
    
    def add_token_mismatch(self, date, time, expected, found, trade_section, expected_tokens, found_tokens, signals):
        self.error_messages += 1
        self.token_mismatch_errors.append({
            "date": date,
            "time": time,
            "expected_count": expected,
            "found_count": found,
            "trade_section": trade_section,
            "expected_tokens": sorted(list(expected_tokens)),
            "found_tokens": sorted(list(found_tokens)),
            "signals": signals,
            "error_timestamp": self.current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "error_type": "token_count_mismatch",
            "user": self.current_user
        })
    
    def add_parsing_error(self, date, time, error_msg, trade_section):
        self.error_messages += 1
        self.parsing_errors.append({
            "date": date,
            "time": time,
            "error": error_msg,
            "trade_section": trade_section,
            "error_timestamp": self.current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "error_type": "parsing_error",
            "user": self.current_user
        })
    
    def generate_report(self):
        return {
            "processing_summary": {
                "total_messages": self.total_messages,
                "successful_messages": self.successful_messages,
                "error_messages": self.error_messages,
                "success_rate": f"{(self.successful_messages/self.total_messages*100):.2f}%" if self.total_messages > 0 else "0%",
                "report_generated": self.current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "generated_by": self.current_user
            },
            "token_mismatch_errors": sorted(self.token_mismatch_errors, key=lambda x: (x["date"], x["time"])),
            "parsing_errors": sorted(self.parsing_errors, key=lambda x: (x["date"], x["time"]))
        }

def normalize_token(token):
    """Normalize token names to their standard form"""
    # Remove common suffixes and clean the token
    token = re.sub(r'(?:USDT|/USDT)$', '', token)
    token = re.sub(r'\((.*?)\)', r'\1', token)
    token = re.sub(r'[-_\s]+', ' ', token)
    token = token.strip().upper()
    
    # Try direct lookup
    if token in TOKEN_LOOKUP:
        return TOKEN_LOOKUP[token]
    
    # Try without spaces
    if token.replace(' ', '') in TOKEN_LOOKUP:
        return TOKEN_LOOKUP[token.replace(' ', '')]
    
    # Try individual parts
    parts = token.split()
    for part in parts:
        if part in TOKEN_LOOKUP:
            return TOKEN_LOOKUP[part]
    
    return token




def extract_signals_from_text(text):
    """Extract trading signals from text using multiple patterns"""
    signals = []
    
    # Clean the text by removing unwanted characters
    cleaned = re.sub(r'[^a-zA-Z0-9.,\-\_\s]', '', text)
    
    # Optionally normalize spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Proceed with existing patterns
    patterns = [
        # Standard format: TOKEN - POSITION x LEVERAGE
        r'(?:\d+\.\s*)?([A-Za-z0-9]+(?:\s*[A-Za-z0-9]*)?)\s*[-–]\s*(Long|Short)(?:\s*[-–]\s*(?:Leverage\s*)?|\s*,\s*|\s+)x(\d+)',
        
        # Position first: POSITION TOKEN x LEVERAGE
        r'(?:\d+\.\s*)?(Long|Short)\s+([A-Za-z0-9]+(?:\s*[A-Za-z0-9]*)?)(?:\s*[-–]\s*(?:Leverage\s*)?)?x(\d+)',
        
        # Dash format: - POSITION TOKEN x LEVERAGE
        r'[-–]\s*(Long|Short)\s+([A-Za-z0-9]+(?:\s*[A-Za-z0-9]*)?)\s+x(\d+)',
        
        # Colon format: TOKEN: POSITION x LEVERAGE
        r'([A-Za-z0-9]+(?:\s*[A-Za-z0-9]*)?):?\s*(Long|Short)(?:\s*[-–]\s*|\s*,\s*|\s+)x(\d+)',
        
        # Parentheses format: TOKEN (POSITION) x LEVERAGE
        r'([A-Za-z0-9]+(?:\s*[A-Za-z0-9]*)?)\s*\((Long|Short)\)\s*x(\d+)',
        
        # Slash format: TOKEN/USDT POSITION x LEVERAGE
        r'([A-Za-z0-9]+(?:/USDT)?)\s*(Long|Short)(?:\s*[-–]\s*|\s*,\s*|\s+)x(\d+)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, cleaned, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            if len(groups) == 3:
                if groups[0].lower() in ['long', 'short']:
                    token = normalize_token(groups[1])
                    position = groups[0].upper()
                    leverage = groups[2]
                else:
                    token = normalize_token(groups[0])
                    position = groups[1].upper()
                    leverage = groups[2]
                
                if token in KNOWN_TOKENS:
                    signal = f"{token}:{position}:{leverage}"
                    signals.append(signal)
    
    return list(dict.fromkeys(signals))


def extract_trading_signals(message, error_tracker):
    """Process a message and extract trading signals"""
    error_tracker.total_messages += 1
    
    if not message.get('trade_section'):
        error_tracker.add_parsing_error(
            message.get('date', ''),
            message.get('time', ''),
            "No trade section found",
            None
        )
        return None
    
    trade_section = message['trade_section']
    signals = extract_signals_from_text(trade_section)
    
    entry = {
        "date": message.get('date', ''),
        "time": message.get('time', ''),
        "trade_section": trade_section,
        "signals": signals,
        "processed_at": error_tracker.current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    
    # Get tokens from the signals
    found_tokens = {signal.split(':')[0] for signal in signals}
    expected_tokens = {normalize_token(token) for token in re.findall(r'\b[A-Z0-9]+\b', trade_section.upper())
                      if normalize_token(token) in KNOWN_TOKENS}
    
    # Compare expected vs found tokens
    if expected_tokens != found_tokens:
        error_tracker.add_token_mismatch(
            entry["date"],
            entry["time"],
            len(expected_tokens),
            len(found_tokens),
            trade_section,
            expected_tokens,
            found_tokens,
            signals
        )
    else:
        error_tracker.successful_messages += 1
    
    return entry if signals else None

def process_messages(input_file='messages.json', output_file='processed_signals.json', report_file='signal_processing_report.json'):
    """Main processing function"""
    error_tracker = SignalProcessingError()
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        
        # Process messages
        processed_signals = []
        for message in messages:
            try:
                processed_entry = extract_trading_signals(message, error_tracker)
                if processed_entry:
                    processed_signals.append(processed_entry)
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                error_tracker.add_parsing_error(
                    message.get('date', ''),
                    message.get('time', ''),
                    str(e),
                    message.get('trade_section', '')
                )
        
# Write processed signals
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_signals, f, indent=2, ensure_ascii=False)
        
        # Generate and write error report
        error_report = error_tracker.generate_report()
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(error_report, f, indent=2, ensure_ascii=False)
        
        # Log summary
        logging.info(f"Processing completed. "
                    f"Total messages: {error_tracker.total_messages}, "
                    f"Successful: {error_tracker.successful_messages}, "
                    f"Errors: {error_tracker.error_messages}")
        
        return error_report
    
    except Exception as e:
        logging.error(f"Fatal error during processing: {e}")
        raise

def validate_files(input_file, output_file, report_file):
    """Validate file paths and permissions"""
    # Check input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Check write permissions for output files
    output_dir = os.path.dirname(output_file) or '.'
    report_dir = os.path.dirname(report_file) or '.'
    
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"No write permission for output directory: {output_dir}")
    
    if not os.access(report_dir, os.W_OK):
        raise PermissionError(f"No write permission for report directory: {report_dir}")

def main():
    """Main entry point"""
    input_file = 'messages.json'
    output_file = 'processed_signals.json'
    report_file = 'signal_processing_report.json'
    
    try:
        # Validate files
        validate_files(input_file, output_file, report_file)
        
        # Process messages
        error_report = process_messages(input_file, output_file, report_file)
        
        # Print summary
        summary = error_report["processing_summary"]
        print(f"\nProcessing Summary:")
        print(f"Total messages: {summary['total_messages']}")
        print(f"Successful: {summary['successful_messages']}")
        print(f"Errors: {summary['error_messages']}")
        print(f"Success rate: {summary['success_rate']}")
        print(f"Report generated: {summary['report_generated']}")
        print(f"Generated by: {summary['generated_by']}")
        
        if error_report["token_mismatch_errors"]:
            print(f"\nFound {len(error_report['token_mismatch_errors'])} token mismatch errors")
        
        if error_report["parsing_errors"]:
            print(f"\nFound {len(error_report['parsing_errors'])} parsing errors")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()