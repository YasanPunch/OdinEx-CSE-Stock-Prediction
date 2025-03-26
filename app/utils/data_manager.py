import os
import sys
import subprocess
import datetime
import pandas as pd
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app/logs/data_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataManager')

# Define constants
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed_data"
SCRAPER_SCRIPT = BASE_DIR / "utils" / "myScraper.py"
TRANSFORM_SCRIPT = BASE_DIR / "utils" / "transform.py"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "logs", exist_ok=True)

def check_data_freshness():
    """Check if today's data has been scraped already"""
    try:
        today = datetime.datetime.now().strftime('%Y%m%d')
        today_file = RAW_DATA_DIR / f"{today}.csv"
        
        if today_file.exists():
            last_modified = datetime.datetime.fromtimestamp(today_file.stat().st_mtime)
            return {
                "fresh": True,
                "last_update": last_modified.strftime('%Y-%m-%d %H:%M:%S'),
                "file": str(today_file)
            }
        else:
            # Check for the most recent file
            files = list(RAW_DATA_DIR.glob("*.csv"))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                last_modified = datetime.datetime.fromtimestamp(latest_file.stat().st_mtime)
                return {
                    "fresh": False,
                    "last_update": last_modified.strftime('%Y-%m-%d %H:%M:%S'),
                    "file": str(latest_file)
                }
            else:
                return {
                    "fresh": False,
                    "last_update": "No data found",
                    "file": None
                }
    except Exception as e:
        logger.error(f"Error checking data freshness: {str(e)}")
        return {
            "fresh": False,
            "last_update": f"Error: {str(e)}",
            "file": None
        }

def run_scraper(force=False):
    """Run the web scraper to fetch today's data"""
    try:
        # Check if it's after 7pm or if force is True
        current_hour = datetime.datetime.now().hour
        if not force and current_hour < 19:  # Before 7pm
            return {
                "success": False,
                "message": f"Scraping only available after 7pm (current hour: {current_hour}). Use force option to override."
            }
        
        # Check if today's data already exists
        freshness = check_data_freshness()
        today = datetime.datetime.now().strftime('%Y%m%d')
        if freshness["fresh"] and not force:
            return {
                "success": False,
                "message": f"Today's data ({today}) already exists. Use force option to override."
            }
        
        # Run the scraper script as a subprocess
        logger.info(f"Running scraper script: {SCRAPER_SCRIPT}")
        result = subprocess.run(
            [sys.executable, str(SCRAPER_SCRIPT)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Scraper failed: {result.stderr}")
            return {
                "success": False,
                "message": f"Scraper failed: {result.stderr}"
            }
        
        logger.info(f"Scraper output: {result.stdout}")
        
        # Verify the new file was created
        new_check = check_data_freshness()
        if new_check["fresh"]:
            return {
                "success": True,
                "message": f"Successfully scraped data. New file: {new_check['file']}"
            }
        else:
            return {
                "success": False,
                "message": "Scraper ran but no new file was detected."
            }
            
    except Exception as e:
        logger.error(f"Error running scraper: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

def run_transformer():
    """Run the transformer script to update processed data"""
    try:
        logger.info(f"Running transformer script: {TRANSFORM_SCRIPT}")
        result = subprocess.run(
            [sys.executable, str(TRANSFORM_SCRIPT), str(RAW_DATA_DIR), str(PROCESSED_DATA_DIR)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Transformer failed: {result.stderr}")
            return {
                "success": False,
                "message": f"Transformer failed: {result.stderr}"
            }
        
        logger.info(f"Transformer output: {result.stdout}")
        
        # Count the number of files created
        processed_files = list(PROCESSED_DATA_DIR.glob("*_historical.csv"))
        return {
            "success": True,
            "message": f"Successfully transformed data. {len(processed_files)} company files updated."
        }
            
    except Exception as e:
        logger.error(f"Error running transformer: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

def get_available_companies():
    """Get list of available companies from processed data"""
    try:
        processed_files = list(PROCESSED_DATA_DIR.glob("*_historical.csv"))
        companies = [file.stem.split('_')[0] for file in processed_files]
        return sorted(companies)
    except Exception as e:
        logger.error(f"Error getting available companies: {str(e)}")
        return []

def get_data_stats():
    """Get statistics about the available data"""
    try:
        raw_files = list(RAW_DATA_DIR.glob("*.csv"))
        processed_files = list(PROCESSED_DATA_DIR.glob("*_historical.csv"))
        
        # Get date range if files exist
        date_range = {"start": "N/A", "end": "N/A"}
        if raw_files:
            dates = [datetime.datetime.strptime(file.stem, '%Y%m%d') for file in raw_files 
                    if file.stem.isdigit() and len(file.stem) == 8]
            
            if dates:
                date_range = {
                    "start": min(dates).strftime('%Y-%m-%d'),
                    "end": max(dates).strftime('%Y-%m-%d')
                }
        
        return {
            "raw_file_count": len(raw_files),
            "processed_file_count": len(processed_files),
            "date_range": date_range,
            "last_update": check_data_freshness()["last_update"]
        }
    except Exception as e:
        logger.error(f"Error getting data stats: {str(e)}")
        return {
            "raw_file_count": 0,
            "processed_file_count": 0,
            "date_range": {"start": "Error", "end": "Error"},
            "last_update": f"Error: {str(e)}"
        }