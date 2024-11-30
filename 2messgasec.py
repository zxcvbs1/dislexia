import re
import json
from datetime import datetime
import logging
import os
from difflib import get_close_matches


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
    """Normalize token names to their standard form using fuzzy matching"""
    # Clean the token
    token_clean = re.sub(r'[^A-Z0-9]', '', token.upper())

    # Direct lookup
    if token_clean in TOKEN_LOOKUP:
        return TOKEN_LOOKUP[token_clean]

    # Fuzzy matching
    close_matches = get_close_matches(token_clean, TOKEN_LOOKUP.keys(), n=1, cutoff=0.8)
    if close_matches:
        return TOKEN_LOOKUP[close_matches[0]]

    return token_clean



def extract_signals_from_text(text):
    import re

    signals = []
    text_upper = text.upper()

    # Remove markdown formatting and special characters
    text_clean = re.sub(r'\*\*|\*|[_`]', '', text_upper)
    text_clean = re.sub(r'[^A-Z0-9\s\:\-\,\.\(\)/]', ' ', text_clean)

    # Flatten KNOWN_TOKENS variants into a mapping
    token_map = {}
    for standard_token, variants in KNOWN_TOKENS.items():
        for variant in variants:
            token_map[variant.upper()] = standard_token

    # Sort variants by length in descending order to replace longer sequences first
    sorted_variants = sorted(token_map.keys(), key=len, reverse=True)

    # Replace all token variants in the text with their standard token names
    for variant in sorted_variants:
        # Use word boundaries and handle special characters within tokens
        pattern = re.escape(variant)
        text_clean = re.sub(r'\b' + pattern + r'\b', token_map[variant], text_clean)

    # Now tokenize the text
    words = re.findall(r'\b\w+\b', text_clean)

    positions = {'LONG', 'SHORT'}

    for i, word in enumerate(words):
        if word in KNOWN_TOKENS:
            token = word
            position = None
            leverage = None

            # Search in a window around the token
            window_size = 5
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            window_words = words[start:end]

            # Find position
            for w in window_words:
                if w in positions:
                    position = w
                    break

            # Find leverage
            for idx, w in enumerate(window_words):
                # Match patterns like 'X10', 'LEVERAGE', '10', etc.
                leverage_match = re.search(r'(?:X|LEVERAGE\s*)?(\d+)', w)
                if leverage_match:
                    leverage = leverage_match.group(1)
                    break
                # Handle cases where 'LEVERAGE' is followed by the number
                elif w == 'LEVERAGE' and idx + 1 < len(window_words):
                    next_word = window_words[idx + 1]
                    leverage_match = re.match(r'X?(\d+)', next_word)
                    if leverage_match:
                        leverage = leverage_match.group(1)
                        break

            if position and leverage:
                signal = f"{token}:{position}:{leverage}"
                signals.append(signal)

    # Remove duplicates while preserving order
    signals = list(dict.fromkeys(signals))
    return signals



def extract_trading_signals(message, error_tracker):
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

    if signals:
        entry = {
            "date": message.get('date', ''),
            "time": message.get('time', ''),
            "trade_section": trade_section,
            "signals": signals,
            "processed_at": error_tracker.current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        error_tracker.successful_messages += 1
        return entry
    else:
        error_tracker.add_token_mismatch(
            message.get('date', ''),
            message.get('time', ''),
            expected_count=0,
            found_count=0,
            trade_section=trade_section,
            expected_tokens=set(),
            found_tokens=set(),
            signals=[]
        )
        return None


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