import pandas as pd
import glob
import os
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_transform.log'
)

def transform_market_data(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read all CSV files in the input folder
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    # Dictionary to store DataFrames for each symbol
    stock_data = {}
    
    # Process each daily dump file
    for filename in sorted(all_files):
        # Try different encodings
        try:
            # Try multiple encodings with error handling
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
                
            for encoding in encodings:
                try:
                    df = pd.read_csv(filename, encoding=encoding)
                    logging.info(f"Successfully read {filename} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                
            if df is None:
                logging.error(f"Failed to read {filename} with any encoding")
                continue
            
            # Convert date from filename if needed
            file_date = os.path.basename(filename).split('.')[0]
            
            for _, row in df.iterrows():
                symbol = str(row['Symbol']).strip()
                
                try:
                    daily_data = {
                        'Date': file_date,
                        'Open': float(row['Open (Rs.)']),
                        'High': float(row['High (Rs.)']),
                        'Low': float(row['Low (Rs.)']),
                        'Close': float(row['**Last Trade (Rs.)']),
                        'Volume': float(row['Share Volume']),
                        'Company': str(row['Company Name']).strip()
                    }
                
                    if symbol not in stock_data:
                        stock_data[symbol] = []
                    stock_data[symbol].append(daily_data)
                    
                except (ValueError, KeyError) as e:
                    logging.warning(f"Error processing row for {symbol} in {filename}: {str(e)}")
                    continue
                       
        except Exception as e:
            logging.error(f"Error processing file {filename}: {str(e)}")
            continue
        
    # Save processed data
    successful_saves = 0
    for symbol, data in stock_data.items():
        try:
            stock_df = pd.DataFrame(data)
            stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y%m%d')
            stock_df = stock_df.sort_values('Date')
            
            output_file = os.path.join(output_folder, f'{symbol}_historical.csv')
            stock_df.to_csv(output_file, index=False)
            successful_saves += 1
            
        except Exception as e:
            logging.error(f"Error saving data for symbol {symbol}: {str(e)}")
    
    return successful_saves

if __name__ == "__main__":
    # Check if input and output folders are provided as arguments
    if len(sys.argv) >= 3:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
    else:
        # Default folders
        input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw_data")
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_data")

    num_stocks = transform_market_data(input_folder, output_folder)
    print(f"Successfully processed and saved {num_stocks} individual stocks")