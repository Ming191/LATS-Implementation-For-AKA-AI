"""
Execution engine for LATS - manages interaction with Java backend.
Handles test execution and cumulative coverage computation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import hashlib
import httpx
from .state import TestCase, ExecutionResult, ConditionInfo
from core.config import settings


@dataclass
class JavaExecutionConfig:
    """Configuration for Java backend connection"""

    base_url: str = "http://localhost:8080"
    timeout: float = 30.0


class ExecutionEngine:
    """
    Execution engine that communicates with Java TestCaseManager.

    Responsibilities:
    - Execute new tests via Java backend
    - Retrieve cumulative coverage for test suite
    - Cache execution results to avoid re-execution

    Java backend manages:
    - Test storage (TestCaseManager)
    - .tp file generation
    - Coverage computation (CoverageManager)
    """

    def __init__(self, config: Optional[JavaExecutionConfig] = None):
        if config is None:
            config = JavaExecutionConfig(
                base_url=settings.java_backend_url, timeout=settings.java_backend_timeout
            )
        self.config = config
        self.client = httpx.AsyncClient(base_url=self.config.base_url, timeout=self.config.timeout)

        # Cache: test_code_hash -> test_name (to avoid duplicate execution)
        self.test_cache: Dict[str, str] = {}

    async def execute_new_test(
        self, function_path: str, test_code: str, test_name: str, existing_test_names: List[str]
    ) -> ExecutionResult:
        """
        Execute a new test and get cumulative coverage for entire suite.

        Flow:
        1. Send new test to Java backend
        2. Java executes test, generates .tp file, saves to TestCaseManager
        3. Java computes cumulative coverage using existing + new test
        4. Return cumulative metrics

        Args:
            function_path: Path to function in Java project (e.g., "src/main.cpp::calculate")
            test_code: C++ test code to execute
            test_name: Unique name for this test
            existing_test_names: Names of tests already in suite

        Returns:
            ExecutionResult with cumulative coverage
        """
        # Check cache (avoid executing identical test code)
        test_hash = self._hash_test_code(test_code)
        if test_hash in self.test_cache:
            cached_name = self.test_cache[test_hash]
            # Test already executed with this exact code
            # Get suite coverage including the cached test
            if cached_name not in existing_test_names:
                suite_names = existing_test_names + [cached_name]
            else:
                suite_names = existing_test_names

            return await self.get_suite_coverage(function_path, suite_names)

        try:
            # Call Java backend
            response = await self.client.post(
                "/api/test-execution/execute-with-suite",
                json={
                    "functionPath": function_path,
                    "testScript": test_code,
                    "testCaseName": test_name,
                    "existingTestNames": existing_test_names,
                    "coverageType": "MCDC",
                },
            )

            response.raise_for_status()
            result = response.json()

            # Cache successful execution
            if result.get("status") == "success":
                self.test_cache[test_hash] = test_name

            # Parse result
            return self._parse_java_response(
                result, new_test_name=test_name, suite_names=existing_test_names + [test_name]
            )

        except httpx.HTTPError as e:
            # Network/HTTP error
            return ExecutionResult(
                new_test_name=test_name,
                new_test_compiled=False,
                error_message=f"HTTP error: {str(e)}",
                suite_test_names=existing_test_names,  # Don't add failed test
            )
        except Exception as e:
            # Other errors
            return ExecutionResult(
                new_test_name=test_name,
                new_test_compiled=False,
                error_message=f"Execution error: {str(e)}",
                suite_test_names=existing_test_names,
            )

    async def get_suite_coverage(
        self, function_path: str, test_names: List[str]
    ) -> ExecutionResult:
        """
        Get cumulative coverage for existing test suite.
        Useful for re-evaluating a node without executing new tests.

        Args:
            function_path: Path to function
            test_names: List of test names in suite

        Returns:
            ExecutionResult with cumulative coverage (no new test)
        """
        try:
            response = await self.client.post(
                "/api/test-execution/get-coverage",
                json={
                    "functionPath": function_path,
                    "testCaseNames": test_names,
                    "coverageType": "MCDC",
                },
            )

            response.raise_for_status()
            result = response.json()

            return self._parse_java_response(
                result,
                new_test_name=None,  # No new test
                suite_names=test_names,
            )

        except Exception as e:
            return ExecutionResult(
                new_test_name="",
                new_test_compiled=True,
                error_message=f"Coverage retrieval error: {str(e)}",
                suite_test_names=test_names,
            )

    def _parse_java_response(
        self, response: Dict, new_test_name: Optional[str], suite_names: List[str]
    ) -> ExecutionResult:
        """
        Parse Java backend response into ExecutionResult.

        Java response format:
        {
            "status": "success" | "failed" | "runtime error",
            "coverage": {
                "statement": {"covered": X, "total": Y, "percentage": Z},
                "branch": {...},
                "mcdc": {...}
            },
            "log": "compilation/execution logs",
            "uncoveredConditions": [
                {"condition": "x > 0", "needTrue": true, "needFalse": false},
                ...
            ]
        }
        """
        status = response.get("status", "error")
        compiled = status == "success"
        error_msg = None if compiled else response.get("log", "Unknown error")

        # Parse coverage
        coverage = response.get("coverage", {})
        mcdc = coverage.get("mcdc", {})
        branch = coverage.get("branch", {})
        statement = coverage.get("statement", {})

        # Parse uncovered conditions
        uncovered = []
        for cond_data in response.get("uncoveredConditions", []):
            uncovered.append(
                ConditionInfo(
                    condition=cond_data.get("condition", ""),
                    need_true=cond_data.get("needTrue", False),
                    need_false=cond_data.get("needFalse", False),
                    parent_decision=cond_data.get("parentDecision"),
                )
            )

        # Parse all conditions (if provided by Java)
        all_conditions = []
        for cond_data in response.get("allConditions", []):
            all_conditions.append(
                ConditionInfo(
                    condition=cond_data.get("condition", ""),
                    need_true=cond_data.get("needTrue", False),
                    need_false=cond_data.get("needFalse", False),
                    parent_decision=cond_data.get("parentDecision"),
                )
            )

        # Infer covered conditions (all - uncovered)
        if all_conditions:
            uncovered_set = set(uncovered)
            covered = [cond for cond in all_conditions if cond not in uncovered_set]
        else:
            # Fallback: estimate from coverage percentage
            # If no explicit list, we can't determine individual conditions
            covered = []

        return ExecutionResult(
            new_test_name=new_test_name or "",
            new_test_compiled=compiled,
            error_message=error_msg,
            suite_test_names=suite_names,
            statement_coverage=statement.get("percentage", 0.0) / 100.0,
            branch_coverage=branch.get("percentage", 0.0) / 100.0,
            mcdc_coverage=mcdc.get("percentage", 0.0) / 100.0,
            conditions_covered=covered,
        )

    def _hash_test_code(self, code: str) -> str:
        """Hash test code for caching"""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    def clear_cache(self):
        """Clear execution cache"""
        self.test_cache.clear()

    async def get_all_conditions(self, function_path: str) -> List[ConditionInfo]:
        """
        Get all conditions (MC/DC pairs) for a function.
        Used to initialize TestState.uncovered_conditions.

        Args:
            function_path: Path to function in Java project

        Returns:
            List of all conditions that need to be covered
        """
        try:
            response = await self.client.post(
                "/api/test-execution/get-conditions",
                json={"functionPath": function_path, "coverageType": "MCDC"},
            )

            response.raise_for_status()
            result = response.json()

            conditions = []
            for cond_data in result.get("conditions", []):
                conditions.append(
                    ConditionInfo(
                        condition=cond_data.get("condition", ""),
                        need_true=cond_data.get("needTrue", False),
                        need_false=cond_data.get("needFalse", False),
                        parent_decision=cond_data.get("parentDecision"),
                    )
                )

            return conditions

        except Exception as e:
            # If can't get conditions, return empty list
            # MCTS will rely on coverage percentage only
            return []
