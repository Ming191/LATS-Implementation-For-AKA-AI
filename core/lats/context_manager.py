"""
Context manager for LATS sessions.
Handles caching of function metadata and learned rules across tree search.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import threading


@dataclass
class SessionContext:
    """
    Context for a single LATS session (one function's test generation).
    Cached to avoid redundant data passing and enable learning.
    """
    
    # Session identification
    session_id: str
    function_signature: str
    function_path: str
    
    # Immutable function metadata (cached once)
    function_code: str  # Actual C++ code (for LLM prompts)
    context: str  # Additional context (enums, structs, etc.)
    
    # Generation parameters
    coverage_target: float
    max_iterations: int
    
    # Learning state (accumulated during search)
    learned_rules: List[str] = field(default_factory=list)
    
    # Token tracking (for budget enforcement)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    max_tokens: int = 100_000
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def add_learned_rule(self, rule: str):
        """Add a learned rule from reflection"""
        if rule and rule not in self.learned_rules:
            self.learned_rules.append(rule)
            self.last_accessed = datetime.now()
    
    def add_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """Track token usage for budget enforcement"""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.last_accessed = datetime.now()
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used in this session"""
        return self.total_prompt_tokens + self.total_completion_tokens
    
    @property
    def tokens_remaining(self) -> int:
        """Tokens remaining in budget"""
        return max(0, self.max_tokens - self.total_tokens)
    
    @property
    def budget_exceeded(self) -> bool:
        """Check if token budget exceeded"""
        return self.total_tokens >= self.max_tokens
    
    def get_age_seconds(self) -> float:
        """Get session age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


class ContextManager:
    """
    Manages SessionContext objects for active LATS sessions.
    
    Features:
    - In-memory cache with TTL
    - Thread-safe access
    - Automatic cleanup of expired sessions
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize context manager.
        
        Args:
            ttl_seconds: Time-to-live for cached contexts (default: 1 hour)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, SessionContext] = {}
        self._lock = threading.Lock()
    
    def get_or_create(
        self,
        session_id: str,
        function_signature: str,
        function_path: str,
        function_code: str,
        context: str,
        coverage_target: float,
        max_iterations: int
    ) -> SessionContext:
        """
        Get existing session context or create new one.
        
        Args:
            session_id: Unique session identifier
            function_signature: Function signature
            function_path: Path in Java project
            function_code: C++ code for LLM
            context: Additional context
            coverage_target: Coverage goal
            max_iterations: Max MCTS iterations
            
        Returns:
            SessionContext (cached or new)
        """
        with self._lock:
            if session_id in self._cache:
                ctx = self._cache[session_id]
                ctx.last_accessed = datetime.now()
                return ctx
            
            # Create new context
            ctx = SessionContext(
                session_id=session_id,
                function_signature=function_signature,
                function_path=function_path,
                function_code=function_code,
                context=context,
                coverage_target=coverage_target,
                max_iterations=max_iterations
            )
            
            self._cache[session_id] = ctx
            return ctx
    
    def get(self, session_id: str) -> Optional[SessionContext]:
        """
        Get existing session context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionContext if exists, None otherwise
        """
        with self._lock:
            ctx = self._cache.get(session_id)
            if ctx:
                ctx.last_accessed = datetime.now()
            return ctx
    
    def update_learned_rules(self, session_id: str, rule: str) -> bool:
        """
        Add learned rule to session.
        
        Args:
            session_id: Session identifier
            rule: Learned rule to add
            
        Returns:
            True if updated, False if session not found
        """
        with self._lock:
            if session_id in self._cache:
                self._cache[session_id].add_learned_rule(rule)
                return True
            return False
    
    def add_token_usage(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> bool:
        """
        Track token usage for session.
        
        Args:
            session_id: Session identifier
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            
        Returns:
            True if updated, False if session not found
        """
        with self._lock:
            if session_id in self._cache:
                self._cache[session_id].add_token_usage(
                    prompt_tokens,
                    completion_tokens
                )
                return True
            return False
    
    def remove(self, session_id: str) -> bool:
        """
        Remove session from cache.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]
                return True
            return False
    
    def cleanup_expired(self):
        """
        Remove expired sessions from cache.
        Called periodically to prevent memory bloat.
        """
        now = datetime.now()
        expired = []
        
        with self._lock:
            for session_id, ctx in self._cache.items():
                age = (now - ctx.last_accessed).total_seconds()
                if age > self.ttl_seconds:
                    expired.append(session_id)
            
            for session_id in expired:
                del self._cache[session_id]
        
        return len(expired)
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of session IDs
        """
        with self._lock:
            return list(self._cache.keys())
    
    def get_stats(self) -> Dict:
        """
        Get statistics about cached sessions.
        
        Returns:
            Dictionary with stats
        """
        with self._lock:
            total_sessions = len(self._cache)
            total_tokens = sum(ctx.total_tokens for ctx in self._cache.values())
            total_rules = sum(len(ctx.learned_rules) for ctx in self._cache.values())
            
            return {
                "total_sessions": total_sessions,
                "total_tokens_used": total_tokens,
                "total_learned_rules": total_rules,
                "ttl_seconds": self.ttl_seconds
            }


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager(ttl_seconds: int = 3600) -> ContextManager:
    """
    Get global context manager instance (singleton).
    
    Args:
        ttl_seconds: TTL for first initialization
        
    Returns:
        ContextManager instance
    """
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(ttl_seconds=ttl_seconds)
    return _context_manager
