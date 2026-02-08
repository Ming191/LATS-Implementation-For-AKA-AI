from typing import List, Set, Optional, Dict
from models.node import Node
from interfaces.strategy import IStrategy
import heapq

class GreedyStrategy(IStrategy):
    def __init__(self, foundation_node: Node):
        self.root = foundation_node
        self.cumulative_covered_ids: Set[int] = set()
        
        if self.root.covered_mcdc_ids:
            self.cumulative_covered_ids.update(self.root.covered_mcdc_ids)
            self.root.is_retained = True

        self.candidate_queue: List[tuple] = []
        
    def add_node(self, node: Node) -> None:
        """
        Add a node to the queue to be considered for expansion.
        """
        heapq.heappush(self.candidate_queue, (-node.score, node))

    def select_next_node(self) -> Optional[Node]:
        """
        Selects the best node from the queue.
        """
        if not self.candidate_queue:
            return None
            
        _, best_node = heapq.heappop(self.candidate_queue)
        return best_node

    def update_node(self, node: Node, covered_ids: Set[int]) -> None:
        """
         Updates a node with new execution results (MCDC feedback).
         Calculates the score based on GLOBAL cumulative coverage.
        """
        node.covered_mcdc_ids = covered_ids
        
        new_items = covered_ids - self.cumulative_covered_ids
        
        node.score = len(new_items)
        
        if node.score > 0:
            self.cumulative_covered_ids.update(new_items)
            node.is_retained = True
            self.add_node(node)
        else:
            pass
