"""
Query Endpoints - API routes for question answering
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from api.models import QueryRequest, QueryResponse
from chains.qa_chain import answer_question, stream_answer
from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_video(request: QueryRequest):
    """
    Ask a question about video content

    Executes complete RAG pipeline using LangChain.
    """
    logger.info(f"Query request: '{request.question[:50]}...'")

    try:
        history = [
            {"question": m.question, "answer": m.answer}
            for m in (request.chat_history or [])
        ]
        result = answer_question(
            question=request.question,
            video_id=request.video_id,
            retriever_type=request.retriever_type,
            include_citations=request.include_citations,
            top_k=request.top_k,
            chat_history=history or None
        )

        response = QueryResponse(**result)

        logger.info(
            f"Query successful: {result.get('retrieved_chunks', 0)} chunks, "
            f"{result.get('duration_seconds', 0):.2f}s"
        )

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/stream")
async def stream_query(request: QueryRequest):
    """
    Stream answer tokens as Server-Sent Events (SSE).

    Returns a text/event-stream response where each event is one of:
      data: {"token": "..."}   — LLM output token
      data: [SOURCES]{...}     — sources after generation
      data: [DONE]             — end of stream
    """
    logger.info(f"Stream query: '{request.question[:50]}...'")

    history = [
        {"question": m.question, "answer": m.answer}
        for m in (request.chat_history or [])
    ]

    return StreamingResponse(
        stream_answer(
            question=request.question,
            video_id=request.video_id,
            retriever_type=request.retriever_type,
            include_citations=request.include_citations,
            top_k=request.top_k,
            chat_history=history or None,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )