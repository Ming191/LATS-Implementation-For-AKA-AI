"""
FastAPI routes for LATS test generation endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import time
import asyncio

from models.lats_models import (
    LATSSearchRequest,
    LATSSearchResponse,
    CoverageDetails,
    SessionInfo,
    HealthCheckResponse,
)
from core.lats.mcts_controller import MCTSController, MCTSConfig
from core.lats.execution_engine import ExecutionEngine
from core.lats.reward import RewardFunction
from core.lats.context_manager import ContextManager, SessionContext
from core.config import settings

# Initialize shared components
context_manager = ContextManager(ttl_seconds=settings.session_ttl_minutes * 60)
router = APIRouter(prefix="/api/v1/lats", tags=["LATS"])


@router.post("/search", response_model=LATSSearchResponse, status_code=status.HTTP_200_OK)
async def search_tests(request: LATSSearchRequest) -> LATSSearchResponse:
    """
    Generate test suite using LATS (Language Agent Tree Search).

    This endpoint initiates an MCTS-based search to generate a test suite
    that achieves the target MC/DC coverage for the specified function.

    Args:
        request: LATSSearchRequest containing function metadata and search parameters

    Returns:
        LATSSearchResponse with test names, coverage metrics, and search statistics

    Raises:
        HTTPException: 400 if request is invalid, 500 if search fails
    """
    start_time = time.time()

    try:
        # Create or retrieve session context
        session_ctx = context_manager.get_or_create(
            session_id=request.session_id,
            function_signature=request.function_signature,
            function_path=request.function_path,
            function_code=request.function_code,
            context=request.context or "",
            coverage_target=request.coverage_target,
            max_iterations=request.max_iterations,
        )

        # Initialize MCTS controller
        execution_engine = ExecutionEngine()
        reward_function = RewardFunction()

        mcts_config = MCTSConfig(
            max_iterations=request.max_iterations,
            exploration_coef=settings.mcts_exploration_coef,
            expansion_k=3,
            min_k=1,
            max_k=settings.mcts_beam_width,
            adaptive_k=True,
            enable_pruning=True,
            prune_threshold=-2.0,
            beam_width=settings.mcts_beam_width,
            coverage_target=request.coverage_target,
            max_no_progress_iters=10,
            verbose=True,  # Enable logging for API calls
        )

        controller = MCTSController(
            execution_engine=execution_engine, reward_function=reward_function, config=mcts_config
        )

        # Run MCTS search
        best_node = await controller.search(session_ctx)

        # Calculate search time
        search_time = time.time() - start_time

        # Count total tests generated (approximate from tree size)
        total_generated = controller.root.visits if controller.root else 0

        # Build response
        response = LATSSearchResponse(
            session_id=session_ctx.session_id,
            status="success",
            test_names=best_node.state.test_case_names,
            final_coverage=best_node.state.current_coverage,
            iterations=controller.iterations,
            total_tests_generated=total_generated,
            total_tests_in_suite=best_node.state.suite_size,
            tokens_used=session_ctx.total_tokens,
            search_time_seconds=round(search_time, 2),
            learned_rules=session_ctx.learned_rules,
            coverage_details=CoverageDetails(
                statement=best_node.state.coverage_details.get("statement", 0.0),
                branch=best_node.state.coverage_details.get("branch", 0.0),
                mcdc=best_node.state.coverage_details.get("mcdc", 0.0),
            ),
        )

        return response

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Search timed out. Try increasing max_iterations or reducing coverage_target.",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Search failed: {str(e)}"
        )


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str) -> SessionInfo:
    """
    Get information about an active session.

    Args:
        session_id: Session identifier

    Returns:
        SessionInfo with session details

    Raises:
        HTTPException: 404 if session not found
    """
    session = context_manager.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired",
        )

    return SessionInfo(
        session_id=session.session_id,
        function_signature=session.function_signature,
        coverage_target=session.coverage_target,
        tokens_used=session.total_tokens,
        tokens_remaining=session.tokens_remaining,
        age_seconds=session.get_age_seconds(),
        learned_rules=session.learned_rules,
    )


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """
    Delete an active session.

    Args:
        session_id: Session identifier

    Returns:
        204 No Content on success

    Raises:
        HTTPException: 404 if session not found
    """
    removed = context_manager.remove(session_id)

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found"
        )


@router.get("/sessions", response_model=dict)
async def list_sessions():
    """
    List all active sessions.

    Returns:
        Dictionary with session statistics
    """
    stats = context_manager.get_stats()
    return {
        "active_sessions": stats["active_sessions"],
        "session_ids": list(context_manager._cache.keys()),
        "total_tokens_used": sum(s.total_tokens for s in context_manager._cache.values()),
    }


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthCheckResponse with service status
    """
    return HealthCheckResponse(status="ok", version="0.1.0", timestamp=datetime.now().isoformat())
