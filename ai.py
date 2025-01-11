from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from IPython.display import Image, display
from langsmith import traceable
from openai import OpenAI
from typing import List
from scraper import *
import nest_asyncio
import random


llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature = '0.5'
)

# complete the code
class State(TypedDict):
    # the question to be send to the LLM
    question: str

    # the answer returned by the LLM
    modded_question: str

    # a list of context is created
    # from the searches
    context: Annotated[list[AnyMessage], add_messages]
    
RAG_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the latest question in the conversation.
If you don't know the answer, just say that you don't know.
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.

Conversation: {conversation}
Context: {context}
Question: {question}
Answer:"""

def get_vector_db_retriever():
    persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")
    embd = OpenAIEmbeddings()

    # If vector store exists, then load it
    if os.path.exists(persist_path):
        vectorstore = SKLearnVectorStore(
            embedding=embd,
            persist_path=persist_path,
            serializer="parquet"
        )
        return vectorstore.as_retriever(lambda_mult=0)

    # Otherwise, index LangSmith documents and create new vector store
    ls_docs_sitemap_loader = SitemapLoader(web_path="https://docs.smith.langchain.com/sitemap.xml")
    ls_docs = ls_docs_sitemap_loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(ls_docs)

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embd,
        persist_path=persist_path,
        serializer="parquet"
    )
    vectorstore.persist()
    return vectorstore.as_retriever(lambda_mult=0)

def generate_answer(state):
    """ Node to generate answer """

    # get context and question from state
    context = state['context']
    question = state['question']

    answer_template = """Modify the given questions context: {question} using this context: {context}"""
    answer_instructions = answer_template.format(question = question, context = context)

    # Now get the model (LLM) to answer
    modded_question = model.invoke(
        [SystemMessage(content = answer_instructions)] +
        [HumanMessage(content = "Please modify the question context and indicate the source of truth.")]
    )

    return {"modded_question" : modded_question}

def search_web(state):
    """ Retrieve docs from web search """
    tavily_search = TavilySearchResults(max_results=3)    # return three search result
    search_docs = tavily_search.invoke(state['question'])

    # format the search result
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in search_docs
        ]
    )

    return {"context" : [formatted_search_docs]}


MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-4o-mini"
APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an assistant for question modding tasks.
Use the following pieces of retrieved questions to mod the latest question in the conversation.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
"""

openai_client = OpenAI()
nest_asyncio.apply()
retriever = get_vector_db_retriever()

"""
retrieve_documents
- Returns documents fetched from a vectorstore based on the user's question
"""
@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)

"""
generate_response
- Calls `call_openai` to generate a model response after formatting inputs
"""
@traceable(run_type="chain")
def generate_response(question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_openai(messages)

"""
call_openai
- Returns the chat completion output from OpenAI
"""
@traceable(run_type="llm")
def call_openai(
    messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0
) -> str:
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

"""
langsmith_rag
- Calls `retrieve_documents` to fetch documents
- Calls `generate_response` to generate a response based on the fetched documents
- Returns the model response
"""
@traceable(run_type="chain")
def langsmith_rag(question: str):
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.choices[0].message.content

try: result = assistant.invoke({"question": str(random.choice(questions))})
except Exception as e: print(e)

# print the answer to the question
print(f"Question: {result['question']} {result['modded_question'].content}")