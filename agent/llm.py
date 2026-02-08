import os

from crewai import LLM
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

deepseek_llm = LLM(
    model='deepseek/deepseek-chat',
    base_url='https://api.deepseek.com',
    api_key=DEEPSEEK_API_KEY,
    temperature=0
)