import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

LENGTH_CHARS = {
    "短篇（約 800 字）":  800,
    "中篇（約 2000 字）": 2000,
    "長篇（約 4000 字）": 4000,
}
