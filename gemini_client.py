import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client=genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(question, context):
    prompt=f"""
               You are an intelligent PDF reading assistant. Answer the users question using ONLY the information provided in the context. If the answer is not available in the context,tell the user that the PDF does not contain sufficient information. Always provide clear and simple explanations.
               CONTEXT:{context} QUESTION:{question}ANSWER:
    """
    response=client.models.generate_content(model="gemini-3.6-flash",contents=prompt)
    return response.text
    