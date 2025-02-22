import logging
import os

# Ensure logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure Logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_info(message):
    """Log informational messages."""
    logging.info(message)

def log_error(error):
    """Log errors."""
    logging.error(error, exc_info=True)

def log_progress(step, details=""):
    """Log progress updates."""
    logging.info(f"Progress: {step} - {details}")