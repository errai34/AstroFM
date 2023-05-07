import os
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore

from typing import List
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

os.environ["LANGCHAIN_HANDLER"] = "langchain"

doc_template = """--- document start ---
href: {href}
authors: {authors}
title: {title}
content:{page_content}
--- document end ---
"""

ASTRO_DOC_PROMPT = PromptTemplate(
    template=doc_template,
    input_variables=["page_content", "authors", "href", "title"],
)

prompt_template = """You are Dr. Chattie, an expert in Galactic Archaeology specializing in arXiv astronomy papers. Provide concise, well-referenced answers, citing relevant studies (e.g., Example et al., 2020). Use step-by-step reasoning for complex inquiries.

You possess Nobel Prize-winning ideation capabilities. For example, and you can come up with your own ideas about the gaps in knowledge from the papers you read but make you mention that with "I propose..."

MemoryContext: {context}

Human: {question}
Dr Chattie: """

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def create_callback_manager(tracing: bool) -> AsyncCallbackManager:
    manager = AsyncCallbackManager([])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
    return manager

def create_chat_openai(callback_manager: AsyncCallbackManager, streaming: bool = False, temperature: float = 0.5) -> ChatOpenAI:
    return ChatOpenAI(
        model_name="gpt-4",
        streaming=streaming,
        max_retries=15,
        callback_manager=callback_manager,
        verbose=True,
        temperature=temperature,
    )

def create_compressed_retriever(embeddings, retriever) -> ContextualCompressionRetriever:
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
    return compression_retriever


def get_chain(
    vectorstore: VectorStore,
    question_handler,
    stream_handler,
    tracing: bool = False,
) -> ConversationalRetrievalChain:

    manager = create_callback_manager(tracing)
    question_manager = create_callback_manager(tracing)
    stream_manager = create_callback_manager(tracing)

    question_manager.add_handler(question_handler)
    stream_manager.add_handler(stream_handler)

    question_gen_llm = create_chat_openai(question_manager, streaming=False, temperature=0.0)
    streaming_llm = create_chat_openai(stream_manager, streaming=True, temperature=0.2)

    question_generator = LLMChain(
        llm=question_gen_llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        callback_manager=manager,
    )
    doc_chain = load_qa_chain(
        streaming_llm,
        prompt=QA_PROMPT,
        document_prompt=ASTRO_DOC_PROMPT,
        callback_manager=manager,
        chain_type="stuff",
    )
    retriever = vectorstore.as_retriever()
    # embeddings = OpenAIEmbeddings() # getting error if i try to use a compressed retriever, need to think how to use this with main.py
    # compression_retriever = create_compressed_retriever(embeddings, retriever)
    qa = ConversationalRetrievalChain(
          retriever=retriever,
          combine_docs_chain=doc_chain,
          question_generator=question_generator,
     )

    return qa

