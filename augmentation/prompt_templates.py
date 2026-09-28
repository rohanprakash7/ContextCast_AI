"""
Prompt Templates for RAG System using LangChain
Contains all prompt templates for answer generation
"""

from typing import List, Dict
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document

from config.logging_config import get_logger

logger = get_logger(__name__)


# ============================================
# Main QA Prompt Template (LangChain)
# ============================================

QA_PROMPT = PromptTemplate(
    template="""You are an AI assistant helping users understand YouTube video content.

Your task is to answer questions based ONLY on the provided context from video transcripts.

Guidelines:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain enough information, say "I don't have enough information in this video to answer that question"
3. Be specific and cite which part of the video the information comes from
4. Keep answers concise but complete
5. Use the same language and terminology as the video
6. Do not make up or infer information not present in the context

Context from video transcript:
{context}

User question: {question}

Answer:""",
    input_variables=["context", "question"]
)


# ============================================
# QA Prompt with Citations (LangChain)
# ============================================

QA_PROMPT_WITH_CITATIONS = PromptTemplate(
    template="""You are an AI assistant helping users understand YouTube video content.

Your task is to answer questions based ONLY on the provided context from video transcripts, and include citations.

Guidelines:
1. Answer ONLY using information from the provided context
2. After each fact or claim, add a citation in brackets like [Chunk 0], [Chunk 1], etc.
3. If the context doesn't contain enough information, say so clearly
4. Be specific and accurate
5. Multiple chunks can support the same point - cite all relevant chunks
6. Keep answers concise but complete

Context (with chunk IDs):
{context}

User question: {question}

Answer (with citations):""",
    input_variables=["context", "question"]
)


# ============================================
# Conversational QA Prompt (with Chat History)
# ============================================

CONVERSATIONAL_QA_PROMPT = PromptTemplate(
    template="""You are an AI assistant helping users understand YouTube video content through conversation.

You have access to the video transcript context and the conversation history.

Guidelines:
1. Answer based ONLY on the provided video transcript context
2. Consider the conversation history for context
3. Maintain consistency with previous answers
4. If asked about something not in the video, say so clearly
5. Be conversational but accurate

Conversation History:
{chat_history}

Video Transcript Context:
{context}

Current Question: {question}

Answer:""",
    input_variables=["chat_history", "context", "question"]
)


# ============================================
# Helper Functions for LangChain Integration
# ============================================

def format_docs_for_prompt(docs: List[Document], include_chunk_ids: bool = False) -> str:
    """
    Format LangChain Documents into context string
    
    Args:
        docs: List of LangChain Document objects from retriever
        include_chunk_ids: Whether to include chunk IDs for citations
    
    Returns:
        Formatted context string ready for prompt
    
    Example:
        >>> retriever = create_simple_retriever(video_id="kY14KfZQ1TI")
        >>> docs = retriever.invoke("What is deep learning?")
        >>> context = format_docs_for_prompt(docs, include_chunk_ids=True)
        >>> # Use in prompt
    """
    if not docs:
        logger.warning("No documents provided for context formatting")
        return "No context available."
    
    context_parts = []
    
    for doc in docs:
        text = doc.page_content
        chunk_id = doc.metadata.get("chunk_id", 0)
        
        if include_chunk_ids:
            context_parts.append(f"[Chunk {chunk_id}]\n{text}")
        else:
            context_parts.append(text)
    
    # Join with double newlines for separation
    context = "\n\n".join(context_parts)
    
    logger.debug(f"Formatted context: {len(docs)} docs, {len(context)} characters")
    
    return context


def create_qa_chain_prompt(
    include_citations: bool = False,
    include_chat_history: bool = False
) -> PromptTemplate:
    """
    Get appropriate prompt template for use case
    
    Args:
        include_citations: Whether to request citations in answer
        include_chat_history: Whether to include conversation history
    
    Returns:
        LangChain PromptTemplate
    
    Example:
        >>> prompt = create_qa_chain_prompt(include_citations=True)
        >>> # Use in chain:
        >>> chain = prompt | llm
    """
    if include_chat_history:
        logger.info("Using conversational QA prompt")
        return CONVERSATIONAL_QA_PROMPT
    elif include_citations:
        logger.info("Using QA prompt with citations")
        return QA_PROMPT_WITH_CITATIONS
    else:
        logger.info("Using standard QA prompt")
        return QA_PROMPT