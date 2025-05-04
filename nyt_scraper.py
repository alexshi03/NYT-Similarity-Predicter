import requests as req
import time
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    key = os.getenv('API_KEY')
