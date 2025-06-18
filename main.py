from langchain_ollama import OllamaLLM

# Load the LLM
llm = OllamaLLM(model="llama3")

# Ask a simple question
response = llm.invoke("what is the capital of jordan")

print(response)
