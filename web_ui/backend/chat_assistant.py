import os
from pathlib import Path
from pydantic import BaseModel

# Load .env file from the backend directory automatically
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, fall back to OS env vars

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class ChatRequest(BaseModel):
    message: str
    prediction_context: str
    confidence: float

class ChatAssistant:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def get_response(self, request: ChatRequest):
        if not self.client:
            return "OpenAI API Key not found. Please set OPENAI_API_KEY environment variable to enable the AI Radiologist Assistant."
            
        system_prompt = f"""You are an expert AI radiologist assistant.
You are helping analyze a bone fracture X-ray. 
Context: The AI model predicted '{request.prediction_context}' with {request.confidence:.1%} confidence.
Answer the user's questions clearly, concisely, and professionally."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.message}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to OpenAI: {str(e)}"
