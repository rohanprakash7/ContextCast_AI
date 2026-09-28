"""
LLM Client using LangChain's ChatOpenAI
Handles answer generation with OpenAI models
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def create_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False
):
    """
    Create LangChain ChatOpenAI instance
    
    Args:
        model: Model name (default: from settings)
        temperature: Sampling temperature (default: 0.2)
        max_tokens: Maximum tokens in response (default: 500)
        streaming: Enable streaming responses
    
    Returns:
        LangChain ChatOpenAI instance
    
    Example:
        >>> llm = create_llm(temperature=0.2)
        >>> response = llm.invoke("What is AI?")
    """
    model = model or settings.OPENAI_CHAT_MODEL
    temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE
    max_tokens = max_tokens or 500
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=streaming
    )
    
    logger.info(
        f"Created LangChain ChatOpenAI: model={model}, "
        f"temperature={temperature}, max_tokens={max_tokens}"
    )
    
    return llm


def create_llm_chain(
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 500
):
    """
    Create LangChain LLM with output parser
    
    Returns a chain: LLM | StrOutputParser
    
    Args:
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max response tokens
    
    Returns:
        LangChain chain (LLM + parser)
    
    Example:
        >>> llm_chain = create_llm_chain()
        >>> response = llm_chain.invoke("What is AI?")
        >>> # Returns string directly (not AIMessage object)
    """
    llm = create_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Add string output parser
    chain = llm | StrOutputParser()
    
    logger.info("Created LLM chain with StrOutputParser")
    
    return chain