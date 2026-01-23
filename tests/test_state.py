"""
Unit tests for LATS core data structures.
"""

import pytest
from core.lats import TestState, TestCase, ConditionInfo, ExecutionResult, TreeNode


class TestConditionInfo:
    def test_condition_equality(self):
        cond1 = ConditionInfo(condition="x > 0", need_true=True, need_false=False)
        cond2 = ConditionInfo(condition="x > 0", need_true=True, need_false=False)
        cond3 = ConditionInfo(condition="y < 10", need_true=True, need_false=False)
        
        assert cond1 == cond2
        assert cond1 != cond3
    
    def test_condition_hash(self):
        cond1 = ConditionInfo(condition="x > 0", need_true=True, need_false=False)
        cond2 = ConditionInfo(condition="x > 0", need_true=True, need_false=False)
        
        # Should be hashable and equal hashes should be equal
        assert hash(cond1) == hash(cond2)
        
        # Can be used in sets
        cond_set = {cond1, cond2}
        assert len(cond_set) == 1


class TestExecutionResult:
    def test_primary_coverage_returns_mcdc(self):
        result = ExecutionResult(
            new_test_name="test_001",
            new_test_compiled=True,
            suite_test_names=["test_001"],
            statement_coverage=0.8,
            branch_coverage=0.7,
            mcdc_coverage=0.6
        )
        
        assert result.primary_coverage == 0.6
    
    def test_failed_when_not_compiled(self):
        result = ExecutionResult(
            new_test_name="test_001",
            new_test_compiled=False,
            error_message="Compilation error",
            suite_test_names=[]
        )
        
        assert result.failed is True
    
    def test_not_failed_when_compiled(self):
        result = ExecutionResult(
            new_test_name="test_001",
            new_test_compiled=True,
            suite_test_names=["test_001"]
        )
        
        assert result.failed is False


class TestTestState:
    def test_is_terminal_when_target_reached(self):
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.95,
            current_coverage=0.96
        )
        
        assert state.is_terminal() is True
    
    def test_is_terminal_when_no_conditions(self):
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.95,
            current_coverage=0.5,
            uncovered_conditions=[]
        )
        
        assert state.is_terminal() is True
    
    def test_not_terminal_when_conditions_remain(self):
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.95,
            current_coverage=0.5,
            uncovered_conditions=[
                ConditionInfo(condition="x > 0", need_true=True)
            ]
        )
        
        assert state.is_terminal() is False
    
    def test_clone_with_new_test(self):
        # Initial state
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.95,
            test_case_names=["test_001"],
            current_coverage=0.3,
            uncovered_conditions=[
                ConditionInfo(condition="x > 0", need_true=True),
                ConditionInfo(condition="y < 10", need_true=True)
            ]
        )
        
        # Execution result
        exec_result = ExecutionResult(
            new_test_name="test_002",
            new_test_compiled=True,
            suite_test_names=["test_001", "test_002"],
            mcdc_coverage=0.5,
            conditions_covered=[
                ConditionInfo(condition="x > 0", need_true=True)
            ]
        )
        
        # Clone
        new_state = state.clone_with_new_test(exec_result)
        
        # Verify
        assert new_state.suite_size == 2
        assert "test_002" in new_state.test_case_names
        assert new_state.current_coverage == 0.5
        assert len(new_state.uncovered_conditions) == 1
        assert new_state.uncovered_conditions[0].condition == "y < 10"
    
    def test_clone_tracks_errors(self):
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.95
        )
        
        exec_result = ExecutionResult(
            new_test_name="test_001",
            new_test_compiled=False,
            error_message="Variable not initialized",
            suite_test_names=[]  # Failed test not added
        )
        
        new_state = state.clone_with_new_test(exec_result)
        
        assert len(new_state.execution_errors) == 1
        assert "not initialized" in new_state.execution_errors[0]
    
    def test_add_learned_rule(self):
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.95
        )
        
        state.add_learned_rule("Always initialize variables")
        state.add_learned_rule("Check NULL pointers")
        state.add_learned_rule("Always initialize variables")  # Duplicate
        
        assert len(state.learned_rules) == 2
        assert "Always initialize variables" in state.learned_rules
    
    def test_suite_size(self):
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.95,
            test_case_names=["test_001", "test_002", "test_003"]
        )
        
        assert state.suite_size == 3
    
    def test_coverage_progress(self):
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.8,
            current_coverage=0.6
        )
        
        assert abs(state.coverage_progress - 0.75) < 0.001  # 0.6 / 0.8
    
    def test_coverage_progress_caps_at_one(self):
        state = TestState(
            function_signature="int calculate(int x)",
            function_path="src/main.cpp::calculate",
            context="",
            coverage_target=0.8,
            current_coverage=0.9
        )
        
        assert state.coverage_progress == 1.0


class TestTreeNode:
    def test_ucb1_unvisited_returns_inf(self):
        state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95
        )
        node = TreeNode(state=state)
        
        assert node.ucb1_score() == float('inf')
    
    def test_ucb1_with_visits(self):
        parent = TreeNode(
            state=TestState(
                function_signature="int calc(int x)",
                function_path="src/main.cpp::calc",
                context="",
                coverage_target=0.95
            )
        )
        parent.visits = 10
        parent.total_reward = 5.0
        
        child = TreeNode(
            state=TestState(
                function_signature="int calc(int x)",
                function_path="src/main.cpp::calc",
                context="",
                coverage_target=0.95
            ),
            parent=parent
        )
        child.visits = 3
        child.total_reward = 2.0
        
        score = child.ucb1_score(exploration_coef=1.414)
        
        # exploitation = 2.0 / 3 = 0.667
        # exploration = 1.414 * sqrt(ln(10) / 3) = 1.414 * 0.877 = 1.240
        # total ≈ 1.907
        assert score > 1.8 and score < 2.0
    
    def test_update_increments_stats(self):
        node = TreeNode(
            state=TestState(
                function_signature="int calc(int x)",
                function_path="src/main.cpp::calc",
                context="",
                coverage_target=0.95
            )
        )
        
        node.update(0.5)
        node.update(0.7)
        
        assert node.visits == 2
        assert node.total_reward == 1.2
        assert node.average_reward() == 0.6
    
    def test_best_child_exploitation(self):
        parent = TreeNode(
            state=TestState(
                function_signature="int calc(int x)",
                function_path="src/main.cpp::calc",
                context="",
                coverage_target=0.95
            )
        )
        
        child1 = TreeNode(state=parent.state)
        child1.visits = 5
        child1.total_reward = 2.0  # avg = 0.4
        
        child2 = TreeNode(state=parent.state)
        child2.visits = 3
        child2.total_reward = 2.1  # avg = 0.7
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        best = parent.best_child(exploration_coef=0.0)
        assert best == child2
    
    def test_is_leaf(self):
        node = TreeNode(
            state=TestState(
                function_signature="int calc(int x)",
                function_path="src/main.cpp::calc",
                context="",
                coverage_target=0.95
            )
        )
        
        assert node.is_leaf() is True
        
        child = TreeNode(state=node.state)
        node.add_child(child)
        
        assert node.is_leaf() is False
    
    def test_get_path_from_root(self):
        root = TreeNode(
            state=TestState(
                function_signature="int calc(int x)",
                function_path="src/main.cpp::calc",
                context="",
                coverage_target=0.95
            )
        )
        
        child = TreeNode(state=root.state)
        root.add_child(child)
        
        grandchild = TreeNode(state=root.state)
        child.add_child(grandchild)
        
        path = grandchild.get_path_from_root()
        
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child
        assert path[2] == grandchild


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
