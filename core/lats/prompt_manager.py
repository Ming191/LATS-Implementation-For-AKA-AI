"""Prompt manager for loading and rendering prompt templates."""

from typing import List, Dict, Optional
from pathlib import Path


class PromptManager:
    """
    Manages prompt templates for test generation.

    Templates are stored in prompts/unified/ directory and rendered with
    function/context-specific data.
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "prompts" / "unified"
        self.templates_dir = Path(templates_dir)
        self._cache = {}

    def _load_template(self, template_name: str) -> str:
        """Load template from file with caching."""
        if template_name not in self._cache:
            template_path = self.templates_dir / f"{template_name}.txt"
            with open(template_path, "r", encoding="utf-8") as f:
                self._cache[template_name] = f.read()
        return self._cache[template_name]

    def render_initialize(
        self,
        function_signature: str,
        function_code: str,
        context: str,
        uncovered_conditions: List,
        learned_rules: List[str],
        k: int = 3,
    ) -> str:
        """
        Render initialize prompt for generating foundation tests.

        Args:
            function_signature: Function signature
            function_code: Full function source code
            context: Additional context (file path, namespace, etc.)
            uncovered_conditions: List of ConditionInfo objects
            learned_rules: List of learned rules
            k: Number of tests to generate

        Returns:
            Rendered prompt string
        """
        template = self._load_template("initialize")

        # Format uncovered conditions
        conditions_str = self._format_conditions(uncovered_conditions[:10])  # Top 10

        # Format learned rules
        rules_str = self._format_learned_rules(learned_rules)

        return template.format(
            function_signature=function_signature,
            function_code=function_code,
            context=context or "No additional context",
            uncovered_conditions=conditions_str,
            learned_rules=rules_str,
            k=k,
        )

    def render_batch(
        self,
        function_signature: str,
        existing_tests: List[str],
        uncovered_conditions: List,
        learned_rules: List[str],
        recent_errors: List[str],
        k: int = 3,
    ) -> str:
        """Render batch expansion prompt."""
        template = self._load_template("expand_batch")

        existing_str = (
            "\n".join(f"- {test}" for test in existing_tests) if existing_tests else "None"
        )
        conditions_str = self._format_conditions(uncovered_conditions)
        rules_str = self._format_learned_rules(learned_rules)
        errors_str = "\n".join(f"- {err}" for err in recent_errors) if recent_errors else "None"

        return template.format(
            function_signature=function_signature,
            existing_tests=existing_str,
            uncovered_conditions=conditions_str,
            learned_rules=rules_str,
            recent_errors=errors_str,
            k=k,
        )

    def render_targeted(
        self,
        function_signature: str,
        target_condition: str,
        context: str,
        similar_tests: List[str],
        failed_attempts: List[str],
        learned_rules: List[str],
    ) -> str:
        """Render targeted expansion prompt for specific condition."""
        template = self._load_template("expand_targeted")

        similar_str = "\n".join(f"- {test}" for test in similar_tests) if similar_tests else "None"
        failed_str = (
            "\n".join(f"- {fail}" for fail in failed_attempts) if failed_attempts else "None"
        )
        rules_str = self._format_learned_rules(learned_rules)

        return template.format(
            function_signature=function_signature,
            target_condition=target_condition,
            context=context or "No additional context",
            similar_tests=similar_str,
            failed_attempts=failed_str,
            learned_rules=rules_str,
        )

    def render_reflection(
        self,
        test_code: str,
        error_message: str,
        target_condition: Optional[str],
        existing_rules: List[str],
    ) -> str:
        """Render reflection prompt for extracting learned rules."""
        template = self._load_template("reflection")

        rules_str = "\n".join(f"- {rule}" for rule in existing_rules) if existing_rules else "None"

        return template.format(
            test_code=test_code,
            error_message=error_message,
            target_condition=target_condition or "Not specified",
            existing_rules=rules_str,
        )

    def _format_conditions(self, conditions: List) -> str:
        """Format list of ConditionInfo objects for prompt."""
        if not conditions:
            return "All conditions covered"

        lines = []
        for i, cond in enumerate(conditions[:20], 1):  # Limit to 20
            if hasattr(cond, "condition"):
                # ConditionInfo object
                need_str = []
                if getattr(cond, "need_true", False):
                    need_str.append("TRUE")
                if getattr(cond, "need_false", False):
                    need_str.append("FALSE")
                needs = " and ".join(need_str) if need_str else "COVERED"
                lines.append(f"{i}. {cond.condition} = {needs}")
            else:
                # String representation
                lines.append(f"{i}. {cond}")

        return "\n".join(lines)

    def _format_learned_rules(self, rules: List[str]) -> str:
        """Format learned rules for prompt."""
        if not rules:
            return "No learned rules yet"
        return "\n".join(f"- {rule}" for rule in rules)
