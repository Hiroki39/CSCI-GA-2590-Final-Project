import openai
import os
from datasets import load_from_disk

openai.api_key = os.environ.get("OPENAI_API_KEY")

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

gsm8k = load_from_disk("Data/processed_gsm8k")