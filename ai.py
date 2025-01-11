from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated
from typing_extensions import TypedDict

from openai import OpenAI

from IPython.display import Image, display

from dotenv import load_dotenv

from scraper import *
from utils import *

import nest_asyncio
import random

load_dotenv()

MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-4o"
MODEL_TEMP = '0.5'
APP_VERSION = 1.0


RAG_PROMPT = """You're an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the latest question in the conversation.
If you don't know the answer, just say that you don't know.
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.

Conversation: {conversation}
Context: {context}
Question: {question}
Answer:"""

RAG_SYSTEM_PROMPT = """You are an assistant for question modding tasks.
Use the following pieces of retrieved questions to mod the latest question in the conversation.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
"""

class State(TypedDict):
    # the question to be send to the LLM
    question: str

    # the answer returned by the LLM
    modded_question: str

    # a list of context is created from the searches
    context: Annotated[list[AnyMessage], add_messages]



# Build the graph

# initialise the state graph
builder = StateGraph(State)

# setup the nodes
builder.add_node("scraping internet", search_web)
builder.add_node("generating question", generate_answer)

# connect the nodes
builder.add_edge(START, "scraping internet")
builder.add_edge("scraping internet", "generating question")
builder.add_edge("generating question", END)

#compile the graph as assistant
assistant = builder.compile()

# display the graph
display(Image(assistant.get_graph().draw_mermaid_png()))

# define model
llm = ChatOpenAI(
    model = MODEL_NAME,
    temperature = MODEL_TEMP
)

openai_client = OpenAI()
nest_asyncio.apply()
retriever = get_vector_db_retriever()

try: result = assistant.invoke({"question": str(random.choice(questions))})
except Exception as e: print(e)

# print the answer to the question
print(f"Question: {result['question']} {result['modded_question'].content}")