from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set your OPENROUTER_API_KEY in file .env")




def get_deepseek(query):
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1",  # OpenRouter model name for DeepSeek-R1
            messages=[
                {"role": "system", "content": "You are a chemistry assistant. Note: Please only output the final answer, no other information and explaination."},
                {"role": "user", "content": query + "Note: Please only output the final answer, no other information and explaination."}
            ],
            stream=False  # Simplified to non-streaming for OpenRouter
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling OpenRouter API for DeepSeek: {e}")
        return "Error"

# print(get_deepseek(''))
class Deepseek():
    name: str = "Deepseek"
    description: str = "Input the question, returns answers using Deepseek model. Note: Please only output the final answer, no other information and explanation."
    def __init__(
        self,
        **tool_args
    ):
        super(Deepseek, self).__init__()

    def _run(self, query: str,**tool_args) -> str:
        return get_deepseek(query)
    
    def __str__(self):
        return "Deepseek"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        try:
            return get_deepseek(query),0
        except:
            return "Error",0
