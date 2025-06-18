from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.stores import InMemoryStore
import uuid

# Step 1: Init LLM
llm = OllamaLLM(model="llama3")

# Step 2: Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{input}")
])

# Step 3: Chain
chain = prompt | llm

# Step 4: Store & Session ID
store = InMemoryStore()
session_id = str(uuid.uuid4())

# Step 5: Memory access
def get_history(session_id):
    existing = store.mget([session_id])[0]
    history = InMemoryChatMessageHistory()
    if existing:
        history.messages = existing
    return history

def set_history(session_id, history):
    store.mset({session_id: history.messages})

# Step 6: Build memory-powered chain
chat_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_history,
    set_session_history=set_history,
    input_messages_key="input"
)

# Step 7: Interactive loop
print("Type 'exit' to quit")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    result = chat_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Bot: {result}")
