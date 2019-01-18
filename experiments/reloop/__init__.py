import logging
from logging import NullHandler
import sys

logging.getLogger(__name__).addHandler(NullHandler())