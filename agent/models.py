from pydantic import BaseModel, Field
from typing import List, Optional

class PlannerOutput(BaseModel):
    """Output for the Planner Agent."""
    target_id: int = Field(..., description="The ID of the uncovered condition to target.")
    condition_text: str = Field(..., description="The text of the condition (e.g., 'x > 10').")
    target_value: bool = Field(..., description="The boolean value we want this condition to evaluate to (True/False).")
    reasoning: str = Field(..., description="Why this condition was chosen.")

class GeneratorOutput(BaseModel):
    """Output for the Generator Agent."""
    test_code: str = Field(..., description="The complete C++ test code snippet.")
    reasoning: str = Field(..., description="Explanation of how the code satisfies the condition.")
