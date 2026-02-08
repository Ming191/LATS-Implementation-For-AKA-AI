from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, Optional, List

@dataclass
class ExecutionFeedback:
    success: bool
    covered_ids: Set[int]
    uncovered_conditions: Optional[List[dict]] = None
    error_message: Optional[str] = None
    execution_logs: Optional[str] = None

class IExecutor(ABC):
    @abstractmethod
    def execute_test(self, test_code: str) -> ExecutionFeedback:
        """
        Executes the given test code and returns feedback.
        """
        pass
