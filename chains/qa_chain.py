"""
QA Chain - Complete RAG Pipeline using LangChain LCEL
Combines retrieval, augmentation, and generation into one chain
"""

import asyncio
import json
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from retrieval import create_retriever
from augmentation.prompt_templates import (
    QA_PROMPT,
    QA_PROMPT_WITH_CITATIONS,
    CONVERSATIONAL_QA_PROMPT,
    format_docs_for_prompt
)
from generation.llm_client import create_llm
from generation.citation_handler import add_source_info
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def create_qa_chain(
    video_id: str,
    retriever_type: str = "simple",
    include_citations: bool = True,
    top_k: int = 4
):
    """
    Create complete QA chain using LangChain LCEL
    
    This chain combines:
    - Retrieval (Phase 6)
    - Augmentation (Phase 7)
    - Generation (Phase 8)
    
    Into one LangChain chain that executes with single invoke()
    
    Args:
        video_id: YouTube video ID to search
        retriever_type: "simple", "rewriting", or "hybrid"
        include_citations: Whether to request citations
        top_k: Number of chunks to retrieve
    
    Returns:
        LangChain LCEL chain
    
    Example:
        >>> chain = create_qa_chain(video_id="kY14KfZQ1TI", retriever_type="simple")
        >>> answer = chain.invoke("What is deep learning?")
        >>> print(answer)  # Just the answer string!
    """
    logger.info(
        f"Creating QA chain: retriever={retriever_type}, "
        f"citations={include_citations}, top_k={top_k}"
    )
    
    # Step 1: Create retriever based on type
    retriever = create_retriever(video_id=video_id, retriever_type=retriever_type, top_k=top_k)
    
    # Step 2: Create context formatting function
    def format_context(docs):
        return format_docs_for_prompt(docs, include_chunk_ids=include_citations)
    
    # Step 3: Select prompt template
    prompt = QA_PROMPT_WITH_CITATIONS if include_citations else QA_PROMPT
    
    # Step 4: Create LLM
    llm = create_llm(temperature=0.2, max_tokens=500)
    
    # Step 5: Build LangChain LCEL chain
    # This is the magic - everything runs automatically!
    qa_chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("Created complete QA chain using LCEL")
    
    return qa_chain


def answer_question(
    question: str,
    video_id: str,
    retriever_type: str = "simple",
    include_citations: bool = True,
    top_k: int = 4,
    chat_history: list = None
) -> Dict[str, Any]:
    """
    Answer a question about a video (complete RAG pipeline)
    
    This is the main function - executes complete pipeline:
    1. Retrieve relevant chunks (Phase 6)
    2. Format into prompt (Phase 7)
    3. Generate answer (Phase 8)
    4. Extract citations (Phase 8)
    
    Args:
        question: User's question
        video_id: Video to search
        retriever_type: Which retrieval strategy
        include_citations: Request citations in answer
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary with answer, citations, sources, and metadata
    
    Example:
        >>> result = answer_question(
        >>>     question="What is deep learning?",
        >>>     video_id="kY14KfZQ1TI",
        >>>     retriever_type="hybrid",
        >>>     include_citations=True
        >>> )
        >>> print(result['answer'])
        >>> print(result['citations'])
    """
    logger.info(f"Answering question: '{question[:50]}...'")
    start_time = datetime.now()
    
    try:
        # Create retriever
        retriever = create_retriever(video_id=video_id, retriever_type=retriever_type, top_k=top_k)
        
        # Retrieve documents
        docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} chunks")
        
        # Format context
        context = format_docs_for_prompt(docs, include_chunk_ids=include_citations)
        
        # Select prompt and format
        if chat_history:
            history_str = "\n".join(
                f"Human: {turn['question']}\nAssistant: {turn['answer']}"
                for turn in chat_history
            )
            prompt_text = CONVERSATIONAL_QA_PROMPT.format(
                chat_history=history_str,
                context=context,
                question=question
            )
        elif include_citations:
            prompt_text = QA_PROMPT_WITH_CITATIONS.format(context=context, question=question)
        else:
            prompt_text = QA_PROMPT.format(context=context, question=question)
        
        # Generate answer
        llm = create_llm(temperature=0.2, max_tokens=500)
        response = llm.invoke(prompt_text)
        answer = response.content
        
        logger.info(f"Generated answer: {len(answer)} characters")
        
        # Process citations if enabled
        if include_citations:
            result = add_source_info(answer, docs)
            result['retrieved_chunks'] = len(docs)
            result['retriever_type'] = retriever_type
        else:
            result = {
                "answer": answer,
                "retrieved_chunks": len(docs),
                "retriever_type": retriever_type
            }
        
        # Add metadata
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result['duration_seconds'] = duration
        result['timestamp'] = end_time.isoformat()
        result['question'] = question
        
        logger.info(f"QA complete in {duration:.2f} seconds")
        
        return result
    
    except Exception as e:
        logger.error(f"QA pipeline failed: {e}")
        raise


async def stream_answer(
    question: str,
    video_id: str,
    retriever_type: str = "simple",
    include_citations: bool = True,
    top_k: int = 4,
    chat_history: list = None
) -> AsyncGenerator[str, None]:
    """
    Stream answer tokens as Server-Sent Events.

    Yields SSE-formatted strings:
      data: {"token": "..."}\n\n   — one per LLM token
      data: [SOURCES]{...}\n\n     — sources payload after generation
      data: [DONE]\n\n             — end signal
    """
    # --- Retrieval (blocking, offloaded to thread) ---
    retriever = create_retriever(video_id=video_id, retriever_type=retriever_type, top_k=top_k)

    docs = await asyncio.to_thread(retriever.invoke, question)
    context = format_docs_for_prompt(docs, include_chunk_ids=include_citations)

    # --- Prompt selection ---
    if chat_history:
        history_str = "\n".join(
            f"Human: {turn['question']}\nAssistant: {turn['answer']}"
            for turn in chat_history
        )
        prompt_text = CONVERSATIONAL_QA_PROMPT.format(
            chat_history=history_str, context=context, question=question
        )
    elif include_citations:
        prompt_text = QA_PROMPT_WITH_CITATIONS.format(context=context, question=question)
    else:
        prompt_text = QA_PROMPT.format(context=context, question=question)

    # --- Stream LLM tokens ---
    llm = create_llm(temperature=0.2, max_tokens=500, streaming=True)
    full_answer = ""

    async for chunk in llm.astream(prompt_text):
        token = chunk.content
        if token:
            full_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"

    # --- Send sources after generation completes ---
    result = add_source_info(full_answer, docs)
    yield f"data: [SOURCES]{json.dumps({'sources': result.get('sources', [])})}\n\n"
    yield "data: [DONE]\n\n"