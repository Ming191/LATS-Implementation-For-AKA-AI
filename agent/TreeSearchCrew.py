from typing import List, Any
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from agent.models import PlannerOutput, GeneratorOutput

from agent.llm import deepseek_llm


@CrewBase
class TreeSearchCrew:
    """TreeSearchCrew class."""
    agents_config: dict = "configs/agents.yaml"
    tasks_config: dict = "configs/tasks.yaml"
    
    agents: List[Agent]
    tasks: List[Task]

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config['planner'],
            llm=deepseek_llm,
        )

    @agent
    def generator(self) -> Agent:
        return Agent(
            config=self.agents_config['generator'],
            llm=deepseek_llm,
        )

    @agent
    def evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluator'],
            llm=deepseek_llm,
        )

    @agent
    def reflector(self) -> Agent:
        return Agent(
            config=self.agents_config['reflector'],
            llm=deepseek_llm,
        )

    @task
    def planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['planning_task'],
            output_pydantic=PlannerOutput
        )

    @task
    def generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['generation_task'],
            output_pydantic=GeneratorOutput
        )

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluation_task'],
        ) # noqa

    @task
    def reflection_task(self) -> Task:
        return Task(
            config=self.tasks_config['reflection_task'],
        ) # noqa

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.planner(),
                self.generator(),
                self.evaluator(),
                self.reflector()
            ],
            tasks=[
                self.planning_task(),
                self.generation_task(),
                self.evaluation_task(),
                self.reflection_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True
        )

    def run_planning(self, inputs: dict) -> Any:
        """Runs only the Planning step."""
        crew = Crew(
            agents=[self.planner()],
            tasks=[self.planning_task()],
            verbose=True
        )
        result = crew.kickoff(inputs=inputs)
        return result

    def run_generation(self, inputs: dict) -> Any:
        """Runs only the Generation step."""
        crew = Crew(
            agents=[self.generator()],
            tasks=[self.generation_task()],
            verbose=True
        )
        result = crew.kickoff(inputs=inputs)
        return result