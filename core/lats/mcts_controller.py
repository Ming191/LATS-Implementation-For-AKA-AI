"""
MCTS Controller for LATS test generation.
Implements selection, expansion, simulation, and backpropagation phases.
"""

from typing import List, Optional, Dict, Any
import asyncio
import random
from dataclasses import dataclass

from .state import TestState, ExecutionResult, ConditionInfo
from .tree import TreeNode
from .execution_engine import ExecutionEngine
from .reward import RewardFunction, compute_reward
from .context_manager import SessionContext
from .llm_client import DeepSeekClient
from .prompt_manager import PromptManager


@dataclass
class MCTSConfig:
    """Configuration for MCTS search"""

    # Tree search parameters
    max_iterations: int = 100
    exploration_coef: float = 1.414  # sqrt(2) for UCB1
    max_depth: int = 50

    # Expansion parameters
    expansion_k: int = 3  # Number of candidate tests per expansion
    min_k: int = 1
    max_k: int = 5
    adaptive_k: bool = True  # Adjust K based on progress

    # Pruning parameters
    enable_pruning: bool = True
    prune_threshold: float = -2.0  # Prune branches with reward < threshold
    beam_width: int = 5  # Keep top-5 children per node

    # Termination conditions
    coverage_target: float = 0.95
    max_no_progress_iters: int = 10  # Stop if no coverage gain for N iterations

    # Debug
    verbose: bool = False


class MCTSController:
    """
    MCTS controller implementing Language Agent Tree Search.

    The four phases:
    1. Selection: Use UCB1 to traverse from root to leaf
    2. Expansion: Generate K candidate tests using LLM
    3. Simulation: Execute candidates and compute rewards
    4. Backpropagation: Update ancestors with best reward
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        reward_function: RewardFunction,
        config: Optional[MCTSConfig] = None,
    ):
        self.execution_engine = execution_engine
        self.reward_function = reward_function
        self.config = config or MCTSConfig()

        # LLM integration
        self.llm_client = DeepSeekClient()
        self.prompt_manager = PromptManager()

        # Search state
        self.root: Optional[TreeNode] = None
        self.best_node: Optional[TreeNode] = None
        self.iterations: int = 0
        self.no_progress_count: int = 0
        self.last_best_coverage: float = 0.0

    async def search(
        self, session_ctx: SessionContext, initial_state: Optional[TestState] = None
    ) -> TreeNode:
        """
        Run MCTS search to generate test suite.

        Args:
            session_ctx: Session context with function metadata
            initial_state: Initial state (if None, create from session)

        Returns:
            Best node found (terminal or best partial solution)
        """
        # Initialize root
        if initial_state is None:
            # Get all conditions from Java
            all_conditions = await self.execution_engine.get_all_conditions(
                session_ctx.function_path
            )

            initial_state = TestState(
                function_signature=session_ctx.function_signature,
                function_path=session_ctx.function_path,
                context=session_ctx.context,
                coverage_target=session_ctx.coverage_target,
                uncovered_conditions=all_conditions,
                learned_rules=session_ctx.learned_rules.copy(),
            )

        self.root = TreeNode(state=initial_state)
        self.best_node = self.root
        self.iterations = 0
        self.no_progress_count = 0
        self.last_best_coverage = 0.0

        if self.config.verbose:
            print(f"Starting MCTS search for {session_ctx.function_signature}")
            print(f"Target coverage: {session_ctx.coverage_target:.1%}")
            print(f"Max iterations: {self.config.max_iterations}")

        # Main MCTS loop
        for iteration in range(self.config.max_iterations):
            self.iterations = iteration + 1

            # Check termination conditions
            if self._should_terminate(session_ctx):
                break

            # MCTS phases
            leaf = self._selection(self.root)

            if leaf is None:
                if self.config.verbose:
                    print(f"Iteration {iteration + 1}: No selectable leaf")
                break

            # Check if leaf is terminal
            if leaf.state.is_terminal():
                if self.config.verbose:
                    print(f"Iteration {iteration + 1}: Reached terminal state")
                self._update_best(leaf)
                break

            # Expand and simulate
            rewards = await self._expansion_and_simulation(leaf, session_ctx)

            if not rewards:
                if self.config.verbose:
                    print(f"Iteration {iteration + 1}: No valid expansions")
                continue

            # Backpropagate best reward
            best_reward = max(rewards)
            self._backpropagation(leaf, best_reward)

            # Update best node
            self._update_best_from_children(leaf)

            # Progress tracking
            if self.best_node.state.current_coverage > self.last_best_coverage:
                self.last_best_coverage = self.best_node.state.current_coverage
                self.no_progress_count = 0

                if self.config.verbose:
                    print(
                        f"Iteration {iteration + 1}: New best coverage: "
                        f"{self.last_best_coverage:.1%} "
                        f"({len(self.best_node.state.test_case_names)} tests)"
                    )
            else:
                self.no_progress_count += 1

        if self.config.verbose:
            print(f"\nSearch completed after {self.iterations} iterations")
            print(f"Best coverage: {self.best_node.state.current_coverage:.1%}")
            print(f"Test suite size: {len(self.best_node.state.test_case_names)}")

        return self.best_node

    def _selection(self, root: TreeNode) -> Optional[TreeNode]:
        """
        Phase 1: Selection
        Traverse tree using UCB1 until reaching expandable leaf.

        Args:
            root: Root node to start from

        Returns:
            Leaf node to expand (or None if no path exists)
        """
        current = root
        depth = 0

        while depth < self.config.max_depth:
            # Terminal node - return it
            if current.state.is_terminal():
                return current

            # Unexpanded node - return for expansion
            if not current.is_fully_expanded():
                return current

            # Fully expanded - select best child
            if not current.children:
                return current

            # Use UCB1 to select child
            current = current.best_child(exploration_coef=self.config.exploration_coef)
            depth += 1

        # Max depth reached
        return current if depth < self.config.max_depth else None

    async def _expansion_and_simulation(
        self, leaf: TreeNode, session_ctx: SessionContext
    ) -> List[float]:
        """
        Phase 2 & 3: Expansion and Simulation
        Generate K candidate tests, execute them, compute rewards.

        Args:
            leaf: Leaf node to expand
            session_ctx: Session context

        Returns:
            List of rewards for each candidate
        """
        # Determine K (number of candidates)
        k = self._adaptive_k(leaf)

        # Generate candidates using LLM (placeholder - will implement with LLM agent)
        candidates = await self._generate_candidates(leaf, session_ctx, k)

        if not candidates:
            return []

        # Execute candidates and compute rewards
        rewards = []
        valid_children = []

        for test_code, test_name in candidates:
            # Execute test
            exec_result = await self.execution_engine.execute_new_test(
                function_path=session_ctx.function_path,
                test_code=test_code,
                test_name=test_name,
                existing_test_names=leaf.state.test_case_names,
            )

            # Create child state
            child_state = leaf.state.clone_with_new_test(exec_result)

            # Compute reward
            reward = self.reward_function.compute(
                old_state=leaf.state, new_state=child_state, exec_result=exec_result
            )

            # Apply pruning
            if self.config.enable_pruning and reward < self.config.prune_threshold:
                if self.config.verbose:
                    print(f"  Pruned candidate {test_name} (reward={reward:.2f})")
                continue

            # Create child node
            child = TreeNode(state=child_state, parent=leaf)
            valid_children.append((child, reward))
            rewards.append(reward)

        # Apply beam search pruning (keep top-K children)
        if self.config.enable_pruning and len(valid_children) > self.config.beam_width:
            valid_children.sort(key=lambda x: x[1], reverse=True)
            valid_children = valid_children[: self.config.beam_width]
            rewards = [r for _, r in valid_children]

        # Add children to leaf
        for child, reward in valid_children:
            leaf.add_child(child)
            child.update(reward)

        return rewards

    def _backpropagation(self, leaf: TreeNode, reward: float):
        """
        Phase 4: Backpropagation
        Update ancestors with reward.

        Args:
            leaf: Leaf node where simulation started
            reward: Reward to backpropagate
        """
        current = leaf.parent

        while current is not None:
            current.update(reward)
            current = current.parent

    def _adaptive_k(self, node: TreeNode) -> int:
        """
        Determine number of candidates to generate (adaptive K).

        Strategy:
        - Early exploration: high K (explore broadly)
        - Good progress: medium K (exploit + explore)
        - Near target: low K (focused exploitation)

        Args:
            node: Current node

        Returns:
            Number of candidates K
        """
        if not self.config.adaptive_k:
            return self.config.expansion_k

        coverage = node.state.current_coverage
        target = node.state.coverage_target
        progress = coverage / target if target > 0 else 0

        if progress < 0.3:
            # Early stage - explore broadly
            return self.config.max_k
        elif progress < 0.7:
            # Mid stage - balanced
            return self.config.expansion_k
        else:
            # Late stage - focused
            return self.config.min_k

    async def _generate_candidates(
        self, node: TreeNode, session_ctx: SessionContext, k: int
    ) -> List[tuple[str, str]]:
        """
        Generate K candidate test cases using LLM.

        Args:
            node: Current node
            session_ctx: Session context
            k: Number of candidates

        Returns:
            List of (test_code, test_name) tuples
        """
        # Determine generation mode and render prompt
        is_initialize = node.state.current_coverage == 0.0
        is_targeted = (
            len(node.state.uncovered_conditions) <= 3 and node.state.current_coverage > 0.5
        )

        # Adaptive temperature based on search progress
        temperature = self._adaptive_temperature(node.state.current_coverage)

        # Get recent errors from failed children
        recent_errors = self._get_recent_errors(node)

        try:
            if is_initialize:
                # Initial foundation tests
                prompt = self.prompt_manager.render_initialize(
                    function_signature=session_ctx.function_signature,
                    function_code=session_ctx.function_code or "",
                    context=session_ctx.context or "",
                    uncovered_conditions=node.state.uncovered_conditions,
                    learned_rules=session_ctx.learned_rules,
                    k=k,
                )
            elif is_targeted and node.state.uncovered_conditions:
                # Targeted generation for specific condition
                target_condition = str(node.state.uncovered_conditions[0])
                prompt = self.prompt_manager.render_targeted(
                    function_signature=session_ctx.function_signature,
                    target_condition=target_condition,
                    context=session_ctx.context or "",
                    similar_tests=[],  # TODO: Fetch test codes from Java if needed
                    failed_attempts=[],  # TODO: Track failed attempts
                    learned_rules=session_ctx.learned_rules,
                )
            else:
                # Batch expansion
                prompt = self.prompt_manager.render_batch(
                    function_signature=session_ctx.function_signature,
                    existing_tests=node.state.test_case_names[-5:],  # Last 5 test names
                    uncovered_conditions=node.state.uncovered_conditions[:10],
                    learned_rules=session_ctx.learned_rules,
                    recent_errors=recent_errors,
                    k=k,
                )

            # Track prompt tokens
            prompt_tokens = len(prompt) // 4  # Rough estimate: 4 chars per token

            # Call LLM
            response = await self.llm_client.generate(
                prompt=prompt, temperature=temperature, max_tokens=2048
            )

            # Track completion tokens
            completion_tokens = len(response) // 4
            session_ctx.add_token_usage(prompt_tokens, completion_tokens)

            # Parse JSON response
            try:
                json_data = self.llm_client.extract_json_from_response(response)

                # Extract test cases from JSON
                # Expected format: {"tests": [{"name": "...", "code": "..."}, ...]}
                # OR: [{"name": "...", "code": "..."}, ...]
                if isinstance(json_data, dict) and "tests" in json_data:
                    tests_list = json_data["tests"]
                elif isinstance(json_data, list):
                    tests_list = json_data
                else:
                    raise ValueError(f"Unexpected JSON structure: {json_data}")

                candidates = []
                for i, test_obj in enumerate(tests_list[:k]):
                    if isinstance(test_obj, dict):
                        test_code = test_obj.get("code", "")
                        test_name = test_obj.get(
                            "name", f"test_{len(node.state.test_case_names) + i + 1:03d}"
                        )
                    else:
                        # Fallback if test_obj is string
                        test_code = str(test_obj)
                        test_name = f"test_{len(node.state.test_case_names) + i + 1:03d}"

                    # Ensure test name is valid
                    if not test_name:
                        test_name = f"test_{len(node.state.test_case_names) + i + 1:03d}"

                    candidates.append((test_code, test_name))

                # Deduplicate candidates (by name since we don't have codes)
                candidates = self._deduplicate_candidates(candidates, node.state.test_case_names)

                return candidates

            except (ValueError, KeyError, IndexError) as e:
                # JSON parsing failed - log and return empty
                if self.config.verbose:
                    print(f"Failed to parse LLM response: {e}")
                    print(f"Response preview: {response[:200]}...")
                return []

        except Exception as e:
            # LLM call failed
            if self.config.verbose:
                print(f"LLM generation error: {e}")
            return []

    def _adaptive_temperature(self, coverage: float) -> float:
        """
        Adjust temperature based on coverage progress.

        Args:
            coverage: Current coverage ratio (0.0 to 1.0)

        Returns:
            Temperature value (0.5 to 0.9)
        """
        if coverage < 0.3:
            # Early exploration - high diversity
            return 0.9
        elif coverage < 0.7:
            # Mid search - balanced
            return 0.7
        else:
            # Late exploitation - focused
            return 0.5

    def _get_recent_errors(self, node: TreeNode, max_errors: int = 5) -> List[str]:
        """
        Extract recent compilation/execution errors from failed children.

        Args:
            node: Current node
            max_errors: Maximum number of errors to return

        Returns:
            List of error messages
        """
        errors = []
        for child in node.children[-10:]:  # Check last 10 children
            # Check if child has execution result with errors
            if hasattr(child, "execution_result") and child.execution_result:
                result = child.execution_result
                if result.status == "COMPILATION_ERROR" and result.log:
                    errors.append(f"Compilation error: {result.log[:200]}")
                elif result.status == "EXECUTION_ERROR" and result.log:
                    errors.append(f"Execution error: {result.log[:200]}")

            if len(errors) >= max_errors:
                break

        return errors

    def _deduplicate_candidates(
        self, candidates: List[tuple[str, str]], existing_names: List[str]
    ) -> List[tuple[str, str]]:
        """
        Remove duplicate test codes and names from candidates.

        Args:
            candidates: List of (test_code, test_name) tuples
            existing_names: List of existing test names in state

        Returns:
            Deduplicated list of candidates
        """
        seen_hashes = set()
        seen_names = set(existing_names)

        unique_candidates = []
        for test_code, test_name in candidates:
            code_hash = hash(test_code.strip())
            # Skip if code or name is duplicate
            if code_hash not in seen_hashes and test_name not in seen_names:
                unique_candidates.append((test_code, test_name))
                seen_hashes.add(code_hash)
                seen_names.add(test_name)

        return unique_candidates

    def _update_best(self, node: TreeNode):
        """Update best node if this node is better"""
        if self.best_node is None:
            self.best_node = node
            return

        # Prefer higher coverage, then fewer tests
        if node.state.current_coverage > self.best_node.state.current_coverage or (
            node.state.current_coverage == self.best_node.state.current_coverage
            and len(node.state.test_case_names) < len(self.best_node.state.test_case_names)
        ):
            self.best_node = node

    def _update_best_from_children(self, node: TreeNode):
        """Update best node from node's children"""
        for child in node.children:
            self._update_best(child)

    def _should_terminate(self, session_ctx: SessionContext) -> bool:
        """
        Check termination conditions.

        Args:
            session_ctx: Session context

        Returns:
            True if search should terminate
        """
        # Coverage target reached
        if self.best_node.state.current_coverage >= session_ctx.coverage_target:
            return True

        # No progress for too long
        if self.no_progress_count >= self.config.max_no_progress_iters:
            if self.config.verbose:
                print(f"Stopping: No progress for {self.no_progress_count} iterations")
            return True

        # Token budget exceeded
        if session_ctx.budget_exceeded:
            if self.config.verbose:
                print("Stopping: Token budget exceeded")
            return True

        return False

    def get_best_path(self) -> List[str]:
        """
        Get test names from best solution path.

        Returns:
            List of test names in order
        """
        if self.best_node is None:
            return []

        return self.best_node.state.test_case_names

    def get_stats(self) -> Dict[str, Any]:
        """
        Get search statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "iterations": self.iterations,
            "best_coverage": self.best_node.state.current_coverage if self.best_node else 0.0,
            "best_suite_size": len(self.best_node.state.test_case_names) if self.best_node else 0,
            "no_progress_count": self.no_progress_count,
            "tree_size": self._count_nodes(self.root) if self.root else 0,
        }

    def _count_nodes(self, node: TreeNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
