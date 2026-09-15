"""
WSGI entrypoint for production application servers (Gunicorn, uWSGI, Waitress).
"""

import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app as application

if __name__ == "__main__":
    application.run()
