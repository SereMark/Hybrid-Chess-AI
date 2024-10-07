import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'gui', 'assets')
DEFAULT_TIME_CONTROL = 10 * 60