import logging
from pathlib import Path

def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a formatter for INFO level
    info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a formatter for WARNING level
    warning_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s  - %(message)s')

    # Create a formatter for CRITICAL level
    critical_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # Determine the log file path relative to the current script's location
    log_file_path = Path(__file__).resolve().parent / 'picasso_logger.log'

    # Create a file handler and set the formatter
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(critical_formatter)  # Set a common formatter for both file handlers

    # Create a stream handler (console) and set the formatter for INFO level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(info_formatter)  # Set INFO level formatter by default

    # Create another stream handler specifically for WARNING level and set the formatter
    warning_stream_handler = logging.StreamHandler()
    warning_stream_handler.setLevel(logging.WARNING)
    warning_stream_handler.setFormatter(warning_formatter)  # Set WARNING level formatter

    # Create another stream handler specifically for CRITICAL level and set the formatter
    critical_stream_handler = logging.StreamHandler()
    critical_stream_handler.setLevel(logging.CRITICAL)
    critical_stream_handler.setFormatter(critical_formatter)  # Set CRITICAL level formatter

    # Add the file handler and stream handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
#    logger.addHandler(warning_stream_handler)
#    logger.addHandler(critical_stream_handler)


# Call the setup function to configure logging when this module is imported
setup_logging()

## Example usage
#logging.debug('This is a debug message')
#logging.info('This is an info message')
#logging.warning('This is a warning message')
#logging.error('This is an error message')
#logging.critical('This is a critical message')
