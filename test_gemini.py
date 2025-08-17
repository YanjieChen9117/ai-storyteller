#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")
response = model.generate_content("Say 'Hello from Gemini!'")
print(response.text)
