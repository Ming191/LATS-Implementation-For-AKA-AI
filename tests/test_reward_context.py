"""
Unit tests for reward function and context manager.
"""

import pytest
from core.lats import (
    TestState, ExecutionResult, ConditionInfo,
    RewardFunction, RewardConfig, compute_reward,
    ContextManager, SessionContext
)
from datetime import datetime, timedelta


class TestRewardFunction:
    def test_positive_reward_for_coverage_gain(self):
        reward_fn = RewardFunction()
        
        old_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.3
        )
        
        new_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.5,
            test_case_names=["test_001"]
        )
        
        exec_result = ExecutionResult(
            new_test_name="test_001",
            new_test_compiled=True,
            suite_test_names=["test_001"],
            mcdc_coverage=0.5
        )
        
        reward = reward_fn.compute(old_state, new_state, exec_result)
        
        # Coverage gain: 0.2 * 10 = 2.0
        # Compile success: +2.0
        # Early bonus: +3.0 (first test)
        # Size penalty: -0.1
        # Total: ~6.9
        assert reward > 6.0 and reward < 8.0
    
    def test_negative_reward_for_compilation_failure(self):
        reward_fn = RewardFunction()
        
        old_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.3
        )
        
        new_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.3,  # No change
            test_case_names=[]  # Failed test not added
        )
        
        exec_result = ExecutionResult(
            new_test_name="test_001",
            new_test_compiled=False,
            error_message="Compilation error",
            suite_test_names=[],
            mcdc_coverage=0.3
        )
        
        reward = reward_fn.compute(old_state, new_state, exec_result)
        
        # Coverage gain: 0
        # Compile failure: -1.0
        # Total: -1.0
        assert reward < 0
    
    def test_early_bonus_for_first_test(self):
        reward_fn = RewardFunction()
        
        old_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.0,
            test_case_names=[]  # No tests yet
        )
        
        new_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.3,
            test_case_names=["test_001"]
        )
        
        exec_result = ExecutionResult(
            new_test_name="test_001",
            new_test_compiled=True,
            suite_test_names=["test_001"],
            mcdc_coverage=0.3
        )
        
        reward = reward_fn.compute(old_state, new_state, exec_result)
        
        # Should include early bonus (3.0)
        assert reward > 5.0
    
    def test_terminal_bonus(self):
        reward_fn = RewardFunction()
        
        state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.97
        )
        
        bonus = reward_fn.compute_terminal_bonus(state)
        
        # Base bonus (5.0) + excess (0.02 * 10 = 0.2)
        assert bonus > 5.0
    
    def test_custom_config(self):
        config = RewardConfig(
            coverage_weight=20.0,  # Double weight
            compile_reward=5.0
        )
        reward_fn = RewardFunction(config)
        
        old_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.3
        )
        
        new_state = TestState(
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            context="",
            coverage_target=0.95,
            current_coverage=0.5,
            test_case_names=["test_001"]
        )
        
        exec_result = ExecutionResult(
            new_test_name="test_001",
            new_test_compiled=True,
            suite_test_names=["test_001"],
            mcdc_coverage=0.5
        )
        
        reward = reward_fn.compute(old_state, new_state, exec_result)
        
        # Coverage gain: 0.2 * 20 = 4.0
        # Compile: +5.0
        # Total should be higher than default
        assert reward > 8.0


class TestContextManager:
    def test_create_session(self):
        ctx_mgr = ContextManager()
        
        ctx = ctx_mgr.get_or_create(
            session_id="test_session",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="int calc(int x) { return x * 2; }",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        assert ctx.session_id == "test_session"
        assert ctx.function_signature == "int calc(int x)"
        assert ctx.total_tokens == 0
    
    def test_get_existing_session(self):
        ctx_mgr = ContextManager()
        
        ctx1 = ctx_mgr.get_or_create(
            session_id="test_session",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="int calc(int x) { return x * 2; }",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        # Get same session
        ctx2 = ctx_mgr.get("test_session")
        
        assert ctx1 is ctx2
    
    def test_add_learned_rule(self):
        ctx_mgr = ContextManager()
        
        ctx_mgr.get_or_create(
            session_id="test_session",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="int calc(int x) { return x * 2; }",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        success = ctx_mgr.update_learned_rules("test_session", "Always initialize variables")
        assert success is True
        
        ctx = ctx_mgr.get("test_session")
        assert len(ctx.learned_rules) == 1
        assert "initialize" in ctx.learned_rules[0]
    
    def test_duplicate_rules_not_added(self):
        ctx_mgr = ContextManager()
        
        ctx_mgr.get_or_create(
            session_id="test_session",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="int calc(int x) { return x * 2; }",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        ctx_mgr.update_learned_rules("test_session", "Rule 1")
        ctx_mgr.update_learned_rules("test_session", "Rule 1")  # Duplicate
        
        ctx = ctx_mgr.get("test_session")
        assert len(ctx.learned_rules) == 1
    
    def test_token_tracking(self):
        ctx_mgr = ContextManager()
        
        ctx_mgr.get_or_create(
            session_id="test_session",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="int calc(int x) { return x * 2; }",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        ctx_mgr.add_token_usage("test_session", prompt_tokens=100, completion_tokens=50)
        ctx_mgr.add_token_usage("test_session", prompt_tokens=200, completion_tokens=75)
        
        ctx = ctx_mgr.get("test_session")
        assert ctx.total_prompt_tokens == 300
        assert ctx.total_completion_tokens == 125
        assert ctx.total_tokens == 425
    
    def test_budget_exceeded(self):
        ctx_mgr = ContextManager()
        
        ctx = ctx_mgr.get_or_create(
            session_id="test_session",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="int calc(int x) { return x * 2; }",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        ctx.max_tokens = 1000
        
        assert ctx.budget_exceeded is False
        
        ctx_mgr.add_token_usage("test_session", prompt_tokens=600, completion_tokens=500)
        
        ctx = ctx_mgr.get("test_session")
        assert ctx.budget_exceeded is True
    
    def test_remove_session(self):
        ctx_mgr = ContextManager()
        
        ctx_mgr.get_or_create(
            session_id="test_session",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="int calc(int x) { return x * 2; }",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        success = ctx_mgr.remove("test_session")
        assert success is True
        
        ctx = ctx_mgr.get("test_session")
        assert ctx is None
    
    def test_get_stats(self):
        ctx_mgr = ContextManager()
        
        ctx_mgr.get_or_create(
            session_id="session1",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="...",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        ctx_mgr.get_or_create(
            session_id="session2",
            function_signature="int foo(int y)",
            function_path="src/main.cpp::foo",
            function_code="...",
            context="",
            coverage_target=0.90,
            max_iterations=50
        )
        
        ctx_mgr.add_token_usage("session1", 100, 50)
        ctx_mgr.update_learned_rules("session1", "Rule 1")
        
        stats = ctx_mgr.get_stats()
        
        assert stats["total_sessions"] == 2
        assert stats["total_tokens_used"] == 150
        assert stats["total_learned_rules"] == 1


class TestSessionContext:
    def test_token_remaining(self):
        ctx = SessionContext(
            session_id="test",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="...",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        ctx.max_tokens = 1000
        ctx.add_token_usage(300, 200)
        
        assert ctx.tokens_remaining == 500
    
    def test_age_calculation(self):
        ctx = SessionContext(
            session_id="test",
            function_signature="int calc(int x)",
            function_path="src/main.cpp::calc",
            function_code="...",
            context="",
            coverage_target=0.95,
            max_iterations=100
        )
        
        # Mock created_at
        ctx.created_at = datetime.now() - timedelta(seconds=60)
        
        age = ctx.get_age_seconds()
        assert age >= 60 and age < 65


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
