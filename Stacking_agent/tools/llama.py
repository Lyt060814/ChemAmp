from openai import OpenAI
import time
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("Please set your OPENROUTER_API_KEY in file .env")


def get_Llama_api(query):
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    
    messages = [
        {'role': 'system', 'content': 'You are a chemistry assistant. Note: Please only output the final answer, no other information and explaination.'},
        {'role': 'user', 'content': query}
    ]
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",  # OpenRouter model name
            messages=messages,
        )
        time.sleep(1)  # Reduced sleep time
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        return "Error"
    

class Llama():
    name: str = "Llama"
    description: str = "Input the question, returns answers using Llama model. Note: Please only output the final answer, no other information and explanation."
    
    def __init__(
        self,
        **tool_args
    ):
        super(Llama, self).__init__()

    def _run(self, query: str,**tool_args) -> str:
        return get_Llama_api(query)
    
    def __str__(self):
        return "Llama"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        return get_Llama_api(query),0
    