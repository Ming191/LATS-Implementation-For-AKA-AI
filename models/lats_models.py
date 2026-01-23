"""
Pydantic models for LATS API requests and responses.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


class CoverageType(str, Enum):
    """Coverage measurement types"""

    STATEMENT = "statement"
    BRANCH = "branch"
    MCDC = "mcdc"


class LATSSearchRequest(BaseModel):
    """
    Request model for LATS test generation search.

    Example:
        {
            "session_id": "session_123",
            "function_signature": "int calculate(int x, int y)",
            "function_path": "src/Calculator.cpp::calculate",
            "function_code": "int calculate(int x, int y) { ... }",
            "context": "namespace Math { ... }",
            "coverage_target": 0.95,
            "max_iterations": 100
        }
    """

    session_id: str = Field(..., description="Unique session identifier")
    function_signature: str = Field(..., description="Function signature (e.g., 'int foo(int x)')")
    function_path: str = Field(
        ..., description="Path to function in project (e.g., 'src/main.cpp::foo')"
    )
    function_code: str = Field(..., description="Complete function source code")
    context: Optional[str] = Field(
        default="", description="Additional context (enums, structs, etc.)"
    )
    coverage_target: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Target coverage ratio (0.0-1.0)"
    )
    max_iterations: int = Field(default=100, ge=1, le=1000, description="Maximum MCTS iterations")
    coverage_type: CoverageType = Field(
        default=CoverageType.MCDC, description="Coverage type (always MCDC)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123456",
                "function_signature": "int calculate(int x, int y)",
                "function_path": "src/Calculator.cpp::calculate",
                "function_code": "int calculate(int x, int y) {\n  if (x > 0 && y < 100) {\n    return x + y;\n  }\n  return 0;\n}",
                "context": "namespace Math { enum Mode { ADD, SUB }; }",
                "coverage_target": 0.95,
                "max_iterations": 100,
                "coverage_type": "mcdc",
            }
        }


class CoverageDetails(BaseModel):
    """Detailed coverage metrics for different coverage types"""

    statement: float = Field(default=0.0, ge=0.0, le=1.0, description="Statement coverage")
    branch: float = Field(default=0.0, ge=0.0, le=1.0, description="Branch coverage")
    mcdc: float = Field(default=0.0, ge=0.0, le=1.0, description="MC/DC coverage")


class LATSSearchResponse(BaseModel):
    """
    Response model for LATS test generation search.

    Example:
        {
            "session_id": "session_123",
            "status": "success",
            "test_names": ["test_001", "test_002", "test_003"],
            "final_coverage": 0.96,
            "iterations": 15,
            "total_tests_generated": 12,
            "total_tests_in_suite": 3,
            "tokens_used": 8542,
            "search_time_seconds": 12.3,
            "learned_rules": ["Rule 1", "Rule 2"],
            "coverage_details": {
                "statement": 1.0,
                "branch": 1.0,
                "mcdc": 0.96
            }
        }
    """

    session_id: str = Field(..., description="Session identifier from request")
    status: str = Field(..., description="Status: success | failed | timeout")
    test_names: List[str] = Field(default_factory=list, description="Test names in final suite")
    final_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Final MC/DC coverage achieved"
    )
    iterations: int = Field(default=0, description="MCTS iterations performed")
    total_tests_generated: int = Field(default=0, description="Total test candidates generated")
    total_tests_in_suite: int = Field(default=0, description="Number of tests in final suite")
    tokens_used: int = Field(default=0, description="Total LLM tokens consumed")
    search_time_seconds: float = Field(default=0.0, description="Total search duration")
    learned_rules: List[str] = Field(
        default_factory=list, description="Rules learned during search"
    )
    coverage_details: CoverageDetails = Field(
        default_factory=CoverageDetails, description="Detailed coverage"
    )
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123456",
                "status": "success",
                "test_names": ["test_001", "test_002", "test_004"],
                "final_coverage": 0.96,
                "iterations": 15,
                "total_tests_generated": 12,
                "total_tests_in_suite": 3,
                "tokens_used": 8542,
                "search_time_seconds": 12.3,
                "learned_rules": [
                    "Negative values for x trigger false branch of x > 0",
                    "Values >= 100 for y trigger false branch of y < 100",
                ],
                "coverage_details": {"statement": 1.0, "branch": 1.0, "mcdc": 0.96},
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""

    status: str = Field(default="ok", description="Service status")
    version: str = Field(default="0.1.0", description="API version")
    timestamp: str = Field(..., description="Current server time")


class SessionInfo(BaseModel):
    """Session information response"""

    session_id: str
    function_signature: str
    coverage_target: float
    tokens_used: int
    tokens_remaining: int
    age_seconds: float
    learned_rules: List[str]
