"""
Tree node representation for MCTS-based LATS.
Implements UCB1 selection and tree traversal logic.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from .state import TestState


class ActionType(str, Enum):
    """Types of actions that can be taken from a node"""
    INITIALIZE = "initialize"  # Generate first test(s)
    EXPAND_BATCH = "expand_batch"  # Generate test for multiple conditions
    EXPAND_TARGETED = "expand_targeted"  # Generate test for specific condition
    REFINE = "refine"  # Refine existing test


@dataclass
class TreeNode:
    """
    Node in the MCTS search tree.
    
    Each node represents a state (test suite + coverage + uncovered conditions).
    Edges represent actions (generating a new test).
    """
    
    # State at this node
    state: TestState
    
    # Tree structure
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    
    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    
    # Metadata
    action: ActionType = ActionType.INITIALIZE
    depth: int = 0
    value_estimate: float = 0.0  # Heuristic estimate (computed once)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0
    
    def is_fully_expanded(self) -> bool:
        """
        Check if node is fully expanded.
        In LATS, a node is fully expanded after one expansion attempt,
        since we generate K candidates in a single expansion.
        """
        return len(self.children) > 0
    
    def is_terminal(self) -> bool:
        """Check if this state is terminal"""
        return self.state.is_terminal()
    
    def ucb1_score(self, exploration_coef: float = 1.414) -> float:
        """
        Compute UCB1 (Upper Confidence Bound) score for node selection.
        
        UCB1 = exploitation + exploration
             = (total_reward / visits) + c * sqrt(ln(parent_visits) / visits)
        
        Args:
            exploration_coef: Exploration coefficient (typically sqrt(2) ≈ 1.414)
            
        Returns:
            UCB1 score (higher = more promising)
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        if self.parent is None or self.parent.visits == 0:
            return self.total_reward / self.visits  # Root has no exploration term
        
        # Exploitation: average reward
        exploitation = self.total_reward / self.visits
        
        # Exploration: uncertainty bonus
        exploration = exploration_coef * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration
    
    def average_reward(self) -> float:
        """Get average reward (exploitation term only)"""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    def update(self, reward: float):
        """
        Update node statistics after simulation.
        
        Args:
            reward: Reward from simulation
        """
        self.visits += 1
        self.total_reward += reward
    
    def add_child(self, child: 'TreeNode'):
        """Add a child node"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def best_child(self, exploration_coef: float = 0.0) -> Optional['TreeNode']:
        """
        Select best child node.
        
        Args:
            exploration_coef: If 0, pure exploitation (best average reward).
                            If > 0, balance exploration and exploitation.
                            
        Returns:
            Best child node, or None if no children
        """
        if not self.children:
            return None
        
        if exploration_coef == 0.0:
            # Pure exploitation: highest average reward
            return max(self.children, key=lambda n: n.average_reward())
        else:
            # UCB1: balance exploration and exploitation
            return max(self.children, key=lambda n: n.ucb1_score(exploration_coef))
    
    def get_path_from_root(self) -> List['TreeNode']:
        """
        Get the path from root to this node.
        
        Returns:
            List of nodes from root to this node (inclusive)
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def __repr__(self) -> str:
        return (f"TreeNode(depth={self.depth}, visits={self.visits}, "
                f"avg_reward={self.average_reward():.3f}, "
                f"coverage={self.state.current_coverage:.2%}, "
                f"children={len(self.children)})")
    
    def tree_summary(self, max_depth: int = 3) -> str:
        """
        Generate a summary of the tree structure.
        
        Args:
            max_depth: Maximum depth to display
            
        Returns:
            String representation of tree structure
        """
        lines = []
        self._tree_summary_recursive(lines, 0, max_depth)
        return "\n".join(lines)
    
    def _tree_summary_recursive(self, lines: List[str], depth: int, max_depth: int):
        """Recursive helper for tree_summary"""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        lines.append(
            f"{indent}├─ {self.action.value}: "
            f"cov={self.state.current_coverage:.2%} "
            f"visits={self.visits} "
            f"avg_reward={self.average_reward():.2f}"
        )
        
        for child in self.children:
            child._tree_summary_recursive(lines, depth + 1, max_depth)
