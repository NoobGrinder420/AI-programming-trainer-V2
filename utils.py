import tempfile
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langsmith import traceable

from typing import List

from scraper import *
import os


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
    """ 
    Node to generate answer 
    """

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
    """ 
    Retrieve docs from web search 
    WIP, will probably not work
    """
    tavily_search = TavilySearchResults(max_results=3)    # return three search result
    search_docs = tavily_search.invoke(state['question'])

    # format the search result
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in search_docs
        ]
    )

    return {"context" : [formatted_search_docs]}

@traceable(run_type="chain")
def retrieve_documents(question: str):
    """
    retrieve_documents
    - Returns documents fetched from a vectorstore based on the user's question
    """
    return retriever.invoke(question)

@traceable(run_type="chain")
def generate_response(question: str, documents):
    """
    generate_response
    - Calls `call_openai` to generate a model response after formatting inputs
    """
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

@traceable(run_type="llm")
def call_openai(
    messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0
) -> str:
    """
    call_openai
    - Returns the chat completion output from OpenAI
    """
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

@traceable(run_type="chain")
def langsmith_rag(question: str):
    """
    langsmith_rag
    - Calls `retrieve_documents` to fetch documents
    - Calls `generate_response` to generate a response based on the fetched documents
    - Returns the model response
    """
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.choices[0].message.content
