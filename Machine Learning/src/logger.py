import logging

def get_logger():
    logging.basicConfig(
        filename="error.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()