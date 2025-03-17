import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default logging mode
LOG_MODE = os.getenv("LOG_MODE", "development").lower()

# Configure logging with custom logger for reasoning
def configure_logging(log_mode=LOG_MODE):
    # Create root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Define formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Create file handler (always logs everything for debugging)
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Set log levels based on mode
    if log_mode == "development":
        root_logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
    else:  # Production mode
        root_logger.setLevel(logging.ERROR)
        console_handler.setLevel(logging.ERROR)
    
    root_logger.addHandler(console_handler)
    
    # Create special reasoning logger that always shows logs
    reasoning_logger = logging.getLogger('reasoning')
    reasoning_logger.setLevel(logging.INFO)
    reasoning_logger.propagate = False  # Don't propagate to root logger
    
    # Create special handler for reasoning logs
    reasoning_handler = logging.StreamHandler()
    reasoning_formatter = logging.Formatter('Reasoning:\n%(message)s\n')
    reasoning_handler.setFormatter(reasoning_formatter)
    reasoning_handler.setLevel(logging.INFO)
    reasoning_logger.addHandler(reasoning_handler)
    
    # Log the current mode
    if log_mode == "development":
        logging.info(f"Logging mode: DEVELOPMENT - showing all logs")
    else:
        logging.error(f"Logging mode: PRODUCTION - showing errors only")

# Initialize logging with default mode
configure_logging()

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not set in environment variables")
    
if not MONGODB_URI:
    logging.error("MONGODB_URI not set in environment variables")