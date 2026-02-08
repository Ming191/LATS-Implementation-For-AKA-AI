from abc import ABC, abstractmethod
from typing import Optional, Set
from models.node import Node

class IStrategy(ABC):
    @abstractmethod
    def select_next_node(self) -> Optional[Node]:
        """
        Selects the next node to expand based on the strategy.
        """
        pass

    @abstractmethod
    def add_node(self, node: Node) -> None:
        """
        Adds a candidate node to the strategy's internal queue/list.
        """
        pass

    @abstractmethod
    def update_node(self, node: Node, covered_ids: Set[int]) -> None:
        """
        Updates a node with execution feedback and adjusts strategy state (e.g., scores, weights).
        """
        pass
