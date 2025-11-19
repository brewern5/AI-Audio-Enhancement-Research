import os
from dotenv import load_dotenv #type: ignore

class Env():
    def __init__(self):
        self._load_env()
    def _load_env(self):
        try:
            load_dotenv()
        except Exception as e:
            print("Could not load ENV! " + e)

    def get_db_uri(self):
        self._load_env()

        return os.getenv('DB_URI')