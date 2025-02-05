from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class GroqLLMModel:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=self.api_key)
    
    def generate_response(self, prompt, model='mixtral-8x7b-32768'):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful financial advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
            
    def handle_error(self, error):
        if "API key" in str(error):
            return "Invalid or missing API key. Please check your GROQ_API_KEY environment variable."
        elif "model" in str(error):
            return "Invalid model specified. Please use 'mixtral-8x7b-32768'."
        else:
            return f"An error occurred: {str(error)}"