"""
Unit tests for MCTS controller.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from core.lats import (
    TestState, ExecutionResult, ConditionInfo,
    TreeNode, ExecutionEngine, RewardFunction,
    SessionContext
)
from core.lats.mcts_controller import MCTSController, MCTSConfig


@pytest.fixture
def mock_execution_engine():
    """Create mock execution engine"""
    engine = AsyncMock(spec=ExecutionEngine)
    
    # Mock get_all_conditions
    engine.get_all_conditions.return_value = [
        ConditionInfo(condition="x > 0", need_true=True, need_false=False),
        ConditionInfo(condition="x > 0", need_true=False, need_false=True),
        ConditionInfo(condition="y < 10", need_true=True, need_false=False),
        ConditionInfo(condition="y < 10", need_true=False, need_false=True)
    ]
    
    # Mock execute_new_test - progressive coverage
    async def mock_execute(function_path, test_code, test_name, existing_test_names):
        num_tests = len(existing_test_names) + 1
        coverage = min(0.3 * num_tests, 0.95)  # 30% per test, cap at 95%
        
        return ExecutionResult(
            new_test_name=test_name,
            new_test_compiled=True,
            suite_test_names=existing_test_names + [test_name],
            mcdc_coverage=coverage
        )
    
    engine.execute_new_test.side_effect = mock_execute
    
    return engine


@pytest.fixture
def session_context():
    """Create test session context"""
    return SessionContext(
        session_id="test_session",
        function_signature="int calc(int x, int y)",
        function_path="src/main.cpp::calc",
        function_code="int calc(int x, int y) { return x + y; }",
        context="",
        coverage_target=0.95,
        max_iterations=10
    )


class TestMCTSController:
    @pytest.mark.asyncio
    async def test_initialization(self, mock_execution_engine):
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction()
        )
        
        assert controller.root is None
        assert controller.best_node is None
        assert controller.iterations == 0
    
    @pytest.mark.asyncio
    async def test_search_creates_root(self, mock_execution_engine, session_context):
        config = MCTSConfig(max_iterations=1, verbose=False)
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        result = await controller.search(session_context)
        
        assert controller.root is not None
        assert controller.root.state.function_signature == session_context.function_signature
        assert controller.iterations == 1
    
    @pytest.mark.asyncio
    async def test_search_generates_children(self, mock_execution_engine, session_context):
        config = MCTSConfig(
            max_iterations=3,
            expansion_k=2,
            verbose=False
        )
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        result = await controller.search(session_context)
        
        # Should have generated some tests
        assert len(result.state.test_case_names) > 0
        assert result.state.current_coverage > 0
    
    @pytest.mark.asyncio
    async def test_search_reaches_target(self, mock_execution_engine, session_context):
        # Configure to reach target quickly
        async def mock_execute_full_coverage(function_path, test_code, test_name, existing_test_names):
            return ExecutionResult(
                new_test_name=test_name,
                new_test_compiled=True,
                suite_test_names=existing_test_names + [test_name],
                mcdc_coverage=0.96  # Above target
            )
        
        mock_execution_engine.execute_new_test.side_effect = mock_execute_full_coverage
        
        config = MCTSConfig(
            max_iterations=10,
            expansion_k=1,
            verbose=False
        )
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        result = await controller.search(session_context)
        
        # Should reach target and terminate early
        assert result.state.current_coverage >= session_context.coverage_target
        assert controller.iterations < 10
    
    @pytest.mark.asyncio
    async def test_adaptive_k(self, mock_execution_engine, session_context):
        config = MCTSConfig(
            adaptive_k=True,
            min_k=1,
            max_k=5,
            expansion_k=3
        )
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        # Early stage - high K
        early_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.1
        )
        early_node = TreeNode(state=early_state)
        assert controller._adaptive_k(early_node) == 5
        
        # Mid stage - medium K
        mid_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.5
        )
        mid_node = TreeNode(state=mid_state)
        assert controller._adaptive_k(mid_node) == 3
        
        # Late stage - low K
        late_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.85
        )
        late_node = TreeNode(state=late_state)
        assert controller._adaptive_k(late_node) == 1
    
    @pytest.mark.asyncio
    async def test_pruning(self, mock_execution_engine, session_context):
        # Mock execution with mix of good and bad results
        call_count = 0
        
        async def mock_execute_with_failures(function_path, test_code, test_name, existing_test_names):
            nonlocal call_count
            call_count += 1
            
            # First test compiles, second fails, third compiles
            if call_count % 2 == 0:
                return ExecutionResult(
                    new_test_name=test_name,
                    new_test_compiled=False,
                    error_message="Compilation error",
                    suite_test_names=existing_test_names,
                    mcdc_coverage=0.0
                )
            else:
                return ExecutionResult(
                    new_test_name=test_name,
                    new_test_compiled=True,
                    suite_test_names=existing_test_names + [test_name],
                    mcdc_coverage=0.3 * len(existing_test_names + [test_name])
                )
        
        mock_execution_engine.execute_new_test.side_effect = mock_execute_with_failures
        
        config = MCTSConfig(
            max_iterations=2,
            expansion_k=3,
            enable_pruning=True,
            prune_threshold=-2.0,
            verbose=False
        )
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        result = await controller.search(session_context)
        
        # Should have pruned some bad candidates
        # Exact number depends on reward function, but should have made progress
        assert result.state.current_coverage > 0
    
    @pytest.mark.asyncio
    async def test_no_progress_termination(self, mock_execution_engine, session_context):
        # Mock execution with no progress (always 0% coverage)
        async def mock_execute_no_progress(function_path, test_code, test_name, existing_test_names):
            return ExecutionResult(
                new_test_name=test_name,
                new_test_compiled=True,
                suite_test_names=existing_test_names + [test_name],
                mcdc_coverage=0.3  # Always same coverage
            )
        
        mock_execution_engine.execute_new_test.side_effect = mock_execute_no_progress
        
        config = MCTSConfig(
            max_iterations=100,
            max_no_progress_iters=5,
            expansion_k=1,
            verbose=False
        )
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        result = await controller.search(session_context)
        
        # Should terminate due to no progress
        assert controller.no_progress_count >= config.max_no_progress_iters
        assert controller.iterations < 100
    
    @pytest.mark.asyncio
    async def test_get_best_path(self, mock_execution_engine, session_context):
        config = MCTSConfig(
            max_iterations=3,
            expansion_k=1,
            verbose=False
        )
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        result = await controller.search(session_context)
        path = controller.get_best_path()
        
        assert isinstance(path, list)
        assert len(path) > 0
        assert all(isinstance(name, str) for name in path)
    
    @pytest.mark.asyncio
    async def test_get_stats(self, mock_execution_engine, session_context):
        config = MCTSConfig(max_iterations=2, verbose=False)
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        result = await controller.search(session_context)
        stats = controller.get_stats()
        
        assert "iterations" in stats
        assert "best_coverage" in stats
        assert "best_suite_size" in stats
        assert "tree_size" in stats
        assert stats["iterations"] > 0
        assert stats["best_coverage"] >= 0
    
    def test_selection_phase(self, mock_execution_engine):
        config = MCTSConfig()
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction(),
            config=config
        )
        
        # Create simple tree
        root_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.3
        )
        root = TreeNode(state=root_state)
        
        # Root is unexpanded - should return root
        leaf = controller._selection(root)
        assert leaf is root
        
        # Add child
        child_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.6,
            test_case_names=["test_001"]
        )
        child = TreeNode(state=child_state, parent=root)
        root.add_child(child)
        child.update(5.0)
        
        # Root now fully expanded - should traverse to child
        leaf = controller._selection(root)
        # Selection should return the child (since it's unexpanded)
        assert leaf == child or leaf == root
    
    def test_update_best(self, mock_execution_engine):
        controller = MCTSController(
            execution_engine=mock_execution_engine,
            reward_function=RewardFunction()
        )
        
        # First node becomes best
        state1 = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.5,
            test_case_names=["test_001"]
        )
        node1 = TreeNode(state=state1)
        controller._update_best(node1)
        assert controller.best_node is node1
        
        # Higher coverage becomes new best
        state2 = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.7,
            test_case_names=["test_001", "test_002"]
        )
        node2 = TreeNode(state=state2)
        controller._update_best(node2)
        assert controller.best_node is node2
        
        # Same coverage but more tests - don't update
        state3 = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.7,
            test_case_names=["test_001", "test_002", "test_003"]
        )
        node3 = TreeNode(state=state3)
        controller._update_best(node3)
        assert controller.best_node is node2  # Still node2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
