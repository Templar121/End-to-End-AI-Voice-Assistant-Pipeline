# query_generation.py
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the LLM (Google Gemini API key needed)
llm = ChatGoogleGenerativeAI(google_api_key="AIzaSyApBdjdYTSWqW6WWd6l1qUQXd551W80kBE", model="gemini-pro")

def generate_response(query):
    print("Generating response...")
    prompt = f"Answer in two sentences: {query}"
    gen = llm.invoke(prompt)
    response = gen.content

    return response.strip()
