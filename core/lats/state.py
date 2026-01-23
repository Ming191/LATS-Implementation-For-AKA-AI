"""
State representation for LATS tree search.
Defines the complete state at any node in the search tree.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class CoverageType(str, Enum):
    """Coverage measurement types"""
    STATEMENT = "statement"
    BRANCH = "branch"
    MCDC = "mcdc"


@dataclass
class ConditionInfo:
    """
    Represents a single condition that needs to be covered.
    Example: In "if (x > 0 && y < 10)", there are 2 conditions.
    """
    condition: str  # The condition expression (e.g., "x > 0")
    need_true: bool = True  # Need to test TRUE outcome
    need_false: bool = True  # Need to test FALSE outcome
    parent_decision: Optional[str] = None  # Parent decision node (for MC/DC)
    
    def __hash__(self):
        return hash((self.condition, self.need_true, self.need_false))
    
    def __eq__(self, other):
        if not isinstance(other, ConditionInfo):
            return False
        return (self.condition == other.condition and 
                self.need_true == other.need_true and
                self.need_false == other.need_false)


@dataclass
class TestCase:
    """
    A single test case - metadata only.
    Actual code and .tp file are managed by Java TestCaseManager.
    """
    test_name: str  # Unique name (e.g., "LATS_test_001")
    description: str
    coverage_target: str  # What this test aims to cover
    
    # Execution results (populated after execution)
    compiled: bool = False
    execution_error: Optional[str] = None
    coverage_contribution: float = 0.0  # How much coverage this test adds to suite


@dataclass
class ExecutionResult:
    """
    Result from executing a test via Java backend.
    Contains CUMULATIVE coverage for the entire suite.
    
    Flow:
    1. Java executes new test → generates .tp file → saves to TestCaseManager
    2. Java computes cumulative coverage using CoverageManager.getCoverageOfMultiTestCaseAtFunctionLevel()
    3. Returns cumulative metrics for entire suite
    """
    # New test info
    new_test_name: str
    new_test_compiled: bool
    error_message: Optional[str] = None
    
    # Suite-level info
    suite_test_names: List[str] = field(default_factory=list)  # All tests including new one
    
    # CUMULATIVE coverage data (entire suite, not just new test)
    statement_coverage: float = 0.0
    branch_coverage: float = 0.0
    mcdc_coverage: float = 0.0
    
    # Conditions covered by ENTIRE SUITE
    conditions_covered: List[ConditionInfo] = field(default_factory=list)
    
    @property
    def failed(self) -> bool:
        """Whether the new test execution failed"""
        return not self.new_test_compiled or self.error_message is not None
    
    @property
    def primary_coverage(self) -> float:
        """Primary coverage metric (always MC/DC in unified approach)"""
        return self.mcdc_coverage


@dataclass
class TestState:
    """
    Complete state representation at a tree node.
    STATELESS design - Java TestCaseManager manages actual test storage.
    Python only tracks: names, coverage metrics, and learning.
    """
    
    # Immutable context (same across all nodes)
    function_signature: str
    function_path: str  # Path in Java project tree (e.g., "src/main.cpp::calculate")
    context: str  # Additional context (enums, structs, etc.)
    coverage_target: float
    
    # Test suite tracking (names only - Java has the actual code)
    test_case_names: List[str] = field(default_factory=list)
    
    # Coverage tracking (cumulative from Java)
    current_coverage: float = 0.0
    coverage_details: Dict[str, float] = field(default_factory=dict)
    
    # Uncovered conditions (decreases as we progress)
    uncovered_conditions: List[ConditionInfo] = field(default_factory=list)
    
    # Execution history (lightweight - only errors for learning)
    execution_errors: List[str] = field(default_factory=list)
    
    # Learned rules (accumulated through reflection)
    learned_rules: List[str] = field(default_factory=list)
    
    def is_terminal(self) -> bool:
        """
        Check if this state is terminal (search should stop).
        Terminal conditions:
        1. Coverage target achieved
        2. No more uncovered conditions
        3. All remaining conditions deemed infeasible
        """
        return (
            self.current_coverage >= self.coverage_target or
            len(self.uncovered_conditions) == 0
        )
    
    def clone_with_new_test(self, exec_result: ExecutionResult) -> 'TestState':
        """
        Create a child state by adding a new test to the suite.
        This is used during tree expansion.
        
        Args:
            exec_result: Execution result containing cumulative coverage for entire suite
            
        Returns:
            New TestState with updated test names and coverage
            
        Note:
            Test code and .tp files are already saved by Java TestCaseManager.
            We only track the test names here.
        """
        # Calculate coverage delta
        coverage_delta = exec_result.primary_coverage - self.current_coverage
        
        # Calculate new uncovered conditions
        new_uncovered = [
            cond for cond in self.uncovered_conditions 
            if cond not in exec_result.conditions_covered
        ]
        
        # Track error if failed
        new_errors = self.execution_errors.copy()
        if exec_result.failed and exec_result.error_message:
            new_errors.append(exec_result.error_message)
        
        # Create new state
        return TestState(
            function_signature=self.function_signature,
            function_path=self.function_path,
            context=self.context,
            coverage_target=self.coverage_target,
            test_case_names=exec_result.suite_test_names,  # Updated list from Java
            current_coverage=exec_result.primary_coverage,  # Cumulative
            coverage_details={
                "statement": exec_result.statement_coverage,
                "branch": exec_result.branch_coverage,
                "mcdc": exec_result.mcdc_coverage
            },
            uncovered_conditions=new_uncovered,
            execution_errors=new_errors,
            learned_rules=self.learned_rules.copy()  # Copy to allow child-specific learning
        )
    
    def add_learned_rule(self, rule: str):
        """Add a learned rule from reflection"""
        if rule and rule not in self.learned_rules:
            self.learned_rules.append(rule)
    
    @property
    def suite_size(self) -> int:
        """Number of tests in current suite"""
        return len(self.test_case_names)
    
    @property
    def coverage_progress(self) -> float:
        """Progress towards coverage target (0.0 to 1.0)"""
        return min(self.current_coverage / self.coverage_target, 1.0)
    
    @property
    def conditions_remaining(self) -> int:
        """Number of conditions still uncovered"""
        return len(self.uncovered_conditions)
    
    def __repr__(self) -> str:
        return (f"TestState(suite_size={self.suite_size}, "
                f"coverage={self.current_coverage:.2%}, "
                f"uncovered={self.conditions_remaining})")
