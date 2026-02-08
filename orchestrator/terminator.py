from typing import Tuple, Set
from logic.greedy_strategy import GreedyStrategy


class SearchTerminator:
    """
    Determines when the search should terminate based on configurable criteria.
    
    Termination conditions:
    1. Max iterations reached
    2. All condition paths covered (100%)
    3. Queue exhausted (no more candidates with new coverage)
    """
    
    def __init__(self, max_iterations: int = 100):
        """
        Initialize terminator.
        
        Args:
            max_iterations: Maximum search iterations allowed
        """
        self.max_iterations = max_iterations
    
    def should_terminate(
        self,
        iteration: int,
        strategy: GreedyStrategy,
        all_possible_paths: Set[str]
    ) -> Tuple[bool, str]:
        """
        Check if search should terminate.
        
        Args:
            iteration: Current iteration number (1-indexed)
            strategy: GreedyStrategy instance with current coverage state
            all_possible_paths: All possible condition paths for the function
        
        Returns:
            (should_stop, reason): Boolean and reason string
        """
        # Criterion 1: Max iterations reached
        if iteration >= self.max_iterations:
            coverage_pct = self._calculate_coverage_percentage(
                strategy.get_coverage_count(),
                len(all_possible_paths)
            )
            return True, (
                f"Max iterations ({self.max_iterations}) reached. "
                f"Final coverage: {coverage_pct:.1f}% "
                f"({strategy.get_coverage_count()}/{len(all_possible_paths)} paths)"
            )
        
       # Criterion 2: All paths covered
        if strategy.global_visited_paths >= all_possible_paths:
            return True, (
                f"Full coverage achieved! All {len(all_possible_paths)} "
                f"condition paths covered in {iteration} iterations."
            )
        
        # Criterion 3: Queue empty (no more candidates)
        if not strategy.candidate_queue:
            covered = strategy.get_coverage_count()
            total = len(all_possible_paths)
            uncovered = total - covered
            coverage_pct = self._calculate_coverage_percentage(covered, total)
            
            return True, (
                f"Search exhausted. No more candidates with new coverage. "
                f"Final coverage: {coverage_pct:.1f}% ({covered}/{total} paths). "
                f"{uncovered} paths remain uncovered."
            )
        
        # Continue searching
        return False, ""
    
    def _calculate_coverage_percentage(self, covered: int, total: int) -> float:
        """Calculate coverage percentage."""
        if total == 0:
            return 100.0
        return (covered / total) * 100.0
