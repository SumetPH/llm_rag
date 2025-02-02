
import sqlite3
import dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Sequence
from pythainlp.tokenize import word_tokenize
from typing_extensions import Annotated, TypedDict
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

file_paths = [
    './file/sso.txt',
]

docs = UnstructuredLoader(
    file_paths,
    chunking_strategy="basic",
    max_characters=5000,
).load()

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

def custom_tokenizer(text):
    return " ".join(word_tokenize(text, engine="newmm"))

tokenized_docs = [
    Document(page_content=custom_tokenizer(doc.page_content), metadata=doc.metadata)
    for doc in docs
]

from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_documents(
    docs,
    embeddings, 
)

model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
        )

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'คุณเป็นผู้ช่วยที่ต้องใช้ข้อมูลที่ให้มา ตอบเป็นภาษาไทย'
        ),
        (
            'human',
            'ข้อมูล: {context}'
        ),
        MessagesPlaceholder(variable_name="messages"),
    ],
)

graph = StateGraph(state_schema=State)

trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=10,
    start_on="human",
    include_system=True,
)

def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    message = trimmed_messages[-1]
    message.pretty_print()

    results = vector_store.similarity_search_with_score(
        query=message.content,
        k=4,
    )

    context = '\n\n'.join(result[0].page_content for result in results)
    # print(context)

    prompt = prompt_template.invoke(
        {'messages': trimmed_messages, 'context': context}
    )

    response = model.invoke(prompt)

    return {"messages": response}


graph.add_edge(START, "model")
graph.add_node("model", call_model)

sqlite3_conn = sqlite3.connect('./db/checkpoints.sqlite', check_same_thread=False)
sqlite3_memory_checkpoint = SqliteSaver(sqlite3_conn)

rag = graph.compile(checkpointer=sqlite3_memory_checkpoint)


app = FastAPI()

class ChatPayload(BaseModel):
    thread_id: str
    question: str

@app.get("/")
def index():
    return {"message": "FastAPI"}

@app.post('/chat')
def chat(payload: ChatPayload):
    config = {"configurable": {"thread_id": payload.thread_id}}

    output = rag.invoke({'messages': payload.question}, config)
    output['messages'][-1].pretty_print()

    return {"message": output['messages'][-1].content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)