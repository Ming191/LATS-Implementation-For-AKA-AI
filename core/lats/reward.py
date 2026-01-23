"""
Reward computation for LATS tree search.
Calculates reward based on coverage progress and test quality.
"""

from dataclasses import dataclass
from typing import Optional
from .state import TestState, ExecutionResult


@dataclass
class RewardConfig:
    """Configuration for reward function weights"""
    coverage_weight: float = 10.0  # High weight for coverage gain
    compile_reward: float = 2.0  # Bonus for successful compilation
    compile_penalty: float = -1.0  # Penalty for compilation failure
    condition_weight: float = 0.5  # Reward per condition covered
    suite_size_penalty: float = -0.1  # Penalty for each test (encourage compact suites)
    early_bonus: float = 3.0  # Large bonus for first successful test


class RewardFunction:
    """
    Computes reward for transitioning from old_state to new_state.
    
    Design principles:
    1. Coverage gain is most important (weighted 10x)
    2. Compilation success/failure has moderate impact
    3. Encourage compact test suites (small penalty per test)
    4. Early success bonus (to bootstrap the search)
    5. Progressive rewards (consistent improvement)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
    
    def compute(
        self,
        old_state: TestState,
        new_state: TestState,
        exec_result: ExecutionResult
    ) -> float:
        """
        Compute reward for state transition.
        
        Args:
            old_state: State before adding new test
            new_state: State after adding new test
            exec_result: Execution result for new test
            
        Returns:
            Total reward (can be negative for bad moves)
        """
        reward = 0.0
        
        # 1. Coverage gain (most important)
        coverage_delta = new_state.current_coverage - old_state.current_coverage
        coverage_reward = coverage_delta * self.config.coverage_weight
        reward += coverage_reward
        
        # 2. Compilation success/failure
        if exec_result.new_test_compiled:
            reward += self.config.compile_reward
        else:
            reward += self.config.compile_penalty
        
        # 3. Conditions covered in this step
        old_uncovered = len(old_state.uncovered_conditions)
        new_uncovered = len(new_state.uncovered_conditions)
        conditions_covered = old_uncovered - new_uncovered
        condition_reward = conditions_covered * self.config.condition_weight
        reward += condition_reward
        
        # 4. Suite size penalty (encourage compact suites)
        size_penalty = len(new_state.test_case_names) * self.config.suite_size_penalty
        reward += size_penalty
        
        # 5. Early progress bonus (first successful test)
        if (len(old_state.test_case_names) == 0 and 
            exec_result.new_test_compiled and
            coverage_delta > 0):
            reward += self.config.early_bonus
        
        # 6. Clip reward to reasonable range
        reward = max(reward, -5.0)  # Don't penalize too harshly
        reward = min(reward, 15.0)  # Cap maximum reward
        
        return reward
    
    def compute_terminal_bonus(self, state: TestState) -> float:
        """
        Compute bonus reward for reaching terminal state.
        
        Args:
            state: Terminal state
            
        Returns:
            Bonus reward if coverage target reached
        """
        if state.current_coverage >= state.coverage_target:
            # Bonus proportional to how much we exceeded target
            excess = state.current_coverage - state.coverage_target
            return 5.0 + (excess * 10.0)
        
        return 0.0
    
    def normalize_reward(self, reward: float, max_seen: float) -> float:
        """
        Normalize reward to [0, 1] range based on max seen reward.
        Useful for comparing rewards across different functions.
        
        Args:
            reward: Raw reward
            max_seen: Maximum reward seen so far
            
        Returns:
            Normalized reward
        """
        if max_seen <= 0:
            return 0.0
        
        return min(reward / max_seen, 1.0)


def compute_reward(
    old_state: TestState,
    new_state: TestState,
    exec_result: ExecutionResult
) -> float:
    """
    Convenience function for computing reward with default config.
    
    Args:
        old_state: State before transition
        new_state: State after transition
        exec_result: Execution result
        
    Returns:
        Reward value
    """
    reward_fn = RewardFunction()
    return reward_fn.compute(old_state, new_state, exec_result)
