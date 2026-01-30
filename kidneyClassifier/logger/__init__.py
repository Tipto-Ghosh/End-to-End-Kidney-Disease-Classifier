import logging
import os
from datetime import datetime
from from_root import from_root

# make the log directory
logs_dir = os.path.join(from_root() , "logs")
os.makedirs(logs_dir , exist_ok = True)

# Log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# create the log file format
log_format = "[%(asctime)s] Line: %(lineno)d | %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# configure basic logging to file
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = log_format,
    datefmt = date_format,
    level = logging.INFO
)

# create console handler to also print logs to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(log_format , datefmt = date_format)
console_handler.setFormatter(formatter)

# add the console handler to the root logger
logging.getLogger().addHandler(console_handler)