from datetime import datetime
import logging
import os
import utils

logs_dir = utils.make_path_abs("logs")

os.makedirs(logs_dir, exist_ok=True)

log_filename = datetime.now().strftime(logs_dir + "/%Y-%m-%d_%H-%M-%S.log")

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Print to console as well
    ]
)

logger_ = logging.getLogger("stt_project")
