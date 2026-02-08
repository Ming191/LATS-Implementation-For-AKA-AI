import json
from typing import List, Dict, Optional
from models.node import Node
from agent.TreeSearchCrew import TreeSearchCrew

class AgentService:
    def __init__(self):
        self.crew_runner = TreeSearchCrew()

    def plan_expansion(self, uncovered_conditions: List[dict], function_code: str) -> Optional[Dict]:
        inputs = {
            "uncovered_conditions": json.dumps(uncovered_conditions, indent=2),
            "function_code": function_code
        }
        crew_result = self.crew_runner.run_planning(inputs)
        
        if hasattr(crew_result, 'pydantic') and crew_result.pydantic:
             return crew_result.pydantic.model_dump()
        
        print(f"Error: No Pydantic output in result: {crew_result}")
        return None

    def generate_code(self, target_goal: Dict, function_code: str, function_signature: str) -> Optional[str]:
        inputs = {
            "target_goal": json.dumps(target_goal),
            "function_signature": function_signature,
            "function_code": function_code
        }
        crew_result = self.crew_runner.run_generation(inputs)
        
        if hasattr(crew_result, 'pydantic') and crew_result.pydantic:
             return crew_result.pydantic.test_code
             
        print(f"Error: No Pydantic output in generator result: {crew_result}")
        return None

    def generate_candidates(self, node: Node, uncovered: List[dict], context: Dict) -> List[Node]:
        candidates = []

        target = self.plan_expansion(uncovered, context.get("function_code", ""))
        if not target:
            return []

        code = self.generate_code(
            target,
            context.get("function_code", ""),
            context.get("function_signature", "")
        )

        if code:
             new_drivers = node.test_drivers + [code]
             child = Node(
                 focal_method=node.focal_method,
                 current_coverage=node.current_coverage,
                 test_drivers=new_drivers,
                 parent=node
             )
             candidates.append(child)

        return candidates
