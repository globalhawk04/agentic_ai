import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Initialize the internet search tool
search_tool = SerperDevTool()

# --- 1. Define Your Specialist Agents ---

# Agent 1: The Reinforcement Learning Researcher
rl_researcher = Agent(
  role='Senior RL Scientist specializing in real-world control systems',
  goal='Analyze the DeepMind fusion paper and extract the core methodology of "reward shaping" and "sim-to-real" transfer.',
  backstory=(
    "You are a deep expert in Reinforcement Learning. You understand the nuances of reward functions, "
    "policy optimization, and the challenges of deploying simulated agents into the physical world. "
    "Your job is to find the 'how' behind the success."
  ),
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
)

# Agent 2: The Cross-Disciplinary Innovator
innovator = Agent(
  role='A creative, multi-disciplinary strategist and founder',
  goal='Take a core technical methodology and propose a bold, novel application for it in a completely different industry.',
  backstory=(
    "You are a systems thinker. You see patterns and connections that others miss. Your talent is in "
    "taking a breakthrough from one field (like nuclear fusion) and seeing its potential to revolutionize another "
    "(like drug discovery or climate modeling)."
  ),
  verbose=True,
  allow_delegation=False
)

# Agent 3: The "Man off the Street" (The Ultimate Sanity Check)
pragmatist = Agent(
    role='A practical, results-oriented businessperson with no AI expertise',
    goal='Critique the proposed new application for its real-world viability. Ask the simple, common-sense questions.',
    backstory=(
        "You are not a scientist. You are grounded in reality. You hear a grand new idea and immediately "
        "think, 'So what? How does this actually make money or solve a real problem for someone?' "
        "You are the ultimate check against techno-optimism and hype."
    ),
    verbose=True,
    allow_delegation=False
)

# --- 2. Create the Tasks ---

research_task = Task(
  description=(
    "Find and analyze the Google DeepMind paper titled 'Towards practical reinforcement learning for tokamak magnetic control'. "
    "Extract and summarize the key techniques they used for 'reward shaping' and 'episode chunking'. "
    "Explain in simple terms why these methods were crucial for their success."
  ),
  expected_output='A bullet-point summary of the core RL techniques and their importance.',
  agent=rl_researcher
)

propose_task = Task(
  description=(
    "Based on the summarized RL techniques, propose ONE novel application for this 'learn-in-simulation-then-deploy' methodology "
    "in a completely different high-stakes industry, such as drug discovery, autonomous surgery, or climate modeling. "
    "Describe the 'synthetic expert' agent that would need to be created and what its 'reward function' might be."
  ),
  expected_output='A 2-paragraph proposal for a new application, detailing the synthetic expert and its goal.',
  agent=innovator
)

critique_task = Task(
    description=(
        "Review the proposed new application. From a purely practical standpoint, what is the single biggest, most obvious flaw or challenge? "
        "Ask the one simple, 'stupid' question that the experts might be overlooking. For example, 'If you simulate a drug on a computer, how do you know it won't have a rare side effect in a real person?' or 'Is the simulator for this new problem even possible to build?'"
    ),
    expected_output='A single, powerful, and pragmatic question that challenges the core assumption of the proposed application.',
    agent=pragmatist
)

# --- 3. Assemble the Crew and Kick It Off ---

# This Crew will run the tasks sequentially
research_crew = Crew(
  agents=[rl_researcher, innovator, pragmatist],
  tasks=[research_task, propose_task, critique_task],
  process=Process.sequential,
  verbose=2
)

result = research_crew.kickoff()

print("\n\n########################")
print("## Final Strategic Brief:")
print("########################\n")
print(result)
