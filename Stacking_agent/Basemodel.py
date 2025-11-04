import os
from openai import OpenAI
from typing import List, Dict
import time
import random
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("Please set your OPENROUTER_API_KEY in file .env")


class ChatModel():
    def __init__(self, model="openai/gpt-4o", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.api_key = OPENROUTER_API_KEY
    def chat(self, prompt: str, history: List[Dict[str, str]], system_prompt: str = 'You are a helpful assistant',stop_word:str='') -> str:
        """
        Get response with the prompt,history and system prompt.

        Args:
            prompt (str)
            history (List[Dict[str, str]])
            system_prompt (str)

        """
        total_tokens =0 
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for entry in history:
            messages.append(entry)
        messages.append({"role": "user", "content": prompt})

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stop = stop_word
            )
        except Exception as e:
            print(e)
            return "Run Again",history
        total_tokens =0
        total_tokens = response.usage.total_tokens
        response = response.choices[0].message.content
            
        history.append({"role": "assistant", "content": response})
        try:
            response = response.replace("Thought:","\033[92mThought:\033[0m")
            response = response.replace("Action:","\033[93mAction:\033[0m")
            response = response.replace("Action Input:","\033[94mAction Input:\033[0m")
        except:
            pass
        try:
            response = response.replace("Final Answer:","\033[91mFinal Answer:\033[0m")
        except:
            pass
        
        return response,total_tokens
    
    

