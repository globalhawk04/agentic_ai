# Multi-Agent Collaboration with CrewAI

This repository showcases the power of multi-agent AI systems using the `crewai` framework. It provides two distinct examples of how to assemble a "crew" of specialized AI agents to collaborate on complex tasks, from technology research to drug discovery.

## 🚀 Overview

The core idea is to move beyond single-prompt interactions with LLMs. With `crewai`, you can create a team of autonomous agents, each with a specific role, goal, and backstory. These agents can work together sequentially or hierarchically to tackle multi-step problems that would be difficult for a single AI to solve.

This repository demonstrates:
-   **Defining Specialized Agents**: Creating agents with unique roles (e.g., "Senior Research Analyst," "Biochemist").
-   **Assigning granular tasks**: Crafting specific tasks and assigning them to the most suitable agent.
-   **Orchestrating Collaboration**: Assembling agents into a `Crew` and kicking off a sequential process to generate a final, synthesized output.
-   **Tool Integration**: Empowering agents with tools, like internet search (`SerperDevTool`), to gather real-time information.

---

## 🔧 How It Works: The `crewai` Framework

The logic of `crewai` is modeled after a real-world team.

1.  **Agents**: These are the individual contributors. You define them with a `role`, `goal`, and `backstory` to give them a clear identity and purpose. You can also equip them with `tools` (e.g., web search, database access).

    ```python
    from crewai import Agent
    from crewai_tools import SerperDevTool

    search_tool = SerperDevTool()

    researcher = Agent(
      role='Senior Research Analyst',
      goal='Uncover groundbreaking technologies',
      backstory='You are a master of the internet...',
      tools=[search_tool] # Give the agent access to the internet
    )
    ```

2.  **Tasks**: These are the specific assignments for your agents. You provide a `description` of what needs to be done and define the `expected_output`.

    ```python
    from crewai import Task

    research_task = Task(
      description='Conduct a comprehensive search on...',
      expected_output='A bullet-point list of key findings and URLs.',
      agent=researcher # Assign the task to the researcher agent
    )
    ```

3.  **Crew**: This is the team that brings it all together. You add your `agents` and `tasks` to the crew and define the `process` (e.g., `Process.sequential`) by which they will collaborate.

    ```python
    from crewai import Crew, Process

    crew = Crew(
      agents=[researcher, analyst],
      tasks=[research_task, analysis_task],
      process=Process.sequential,
      verbose=2
    )
    ```

---

## 📂 Examples in This Repository

This repository includes two powerful examples to get you started.

### Example 1: Tech Trend Analysis 🤖

This script creates a two-agent crew to research and report on an emerging technology standard called "Agent Gateway Protocols (AGP)."

-   **Agents**:
    1.  `Senior Research Analyst`: Scours the internet for the latest information on AGP.
    2.  `Principal Technology Strategist`: Synthesizes the researcher's findings into a concise, strategic summary for a business audience.
-   **Usage**: Ideal for market research, competitive analysis, or technology forecasting.

#### Code (`tech_research_crew.py`)

```python
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# You can use a different model provider if you wish
# os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"

# Initialize the tool for internet searches
search_tool = SerperDevTool()

# --- 1. Define Your Agents ---

# Agent 1: The Researcher
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover groundbreaking technologies and trends in a specified field',
  backstory=(
    "You are a master of the internet, capable of finding the most relevant "
    "and up-to-date information from news articles, academic papers, and technical blogs. "
    "Your specialty is identifying the signal in the noise."
  ),
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
)

# Agent 2: The Analyst & Writer
analyst = Agent(
  role='Principal Technology Strategist',
  goal='Synthesize complex research findings into a clear, concise, and actionable report',
  backstory=(
    "You are an expert analyst with a knack for storytelling. You take raw data and technical "
    "jargon and transform it into insightful, strategic narratives that a business leader "
    "can understand and act upon."
  ),
  verbose=True,
  allow_delegation=False
)

# --- 2. Create the Tasks ---

# Task for the Researcher
research_task = Task(
  description=(
    "Conduct a comprehensive search on the latest advancements in 'Agent Gateway Protocols (AGP)'. "
    "Focus on what they are, why they are important for the future of AI, and find the key players or open standards "
    "like the a2a-project on GitHub."
  ),
  expected_output='A bullet-point list of key findings, URLs, and relevant snippets. Focus on facts and sources.',
  agent=researcher
)

# Task for the Analyst, which depends on the researcher's output
analysis_task = Task(
  description=(
    "Using the research findings provided, write a 3-paragraph summary report. "
    "The report should cover: \n"
    "1. What Agent Gateway Protocols are and the problem they solve. \n"
    "2. The strategic importance of AGP for the future 'Agentic Web'. \n"
    "3. A concluding thought on why companies should be paying attention to this emerging standard."
  ),
  expected_output='A polished, well-structured 3-paragraph report formatted in markdown.',
  agent=analyst
)

# --- 3. Assemble the Crew ---

# Create the Crew with a sequential process
crew = Crew(
  agents=[researcher, analyst],
  tasks=[research_task, analysis_task],
  process=Process.sequential, # The tasks will be executed one after the other
  verbose=2 # Set to 2 for detailed, step-by-step logging
)

# --- 4. Kick Off the Mission! ---
result = crew.kickoff()

print("\n\n########################")
print("## Final Report:")
print("########################\n")
print(result)```

### Example 2: Drug Discovery & Critique 🧬

This script demonstrates a more complex, three-agent crew designed to simulate a simplified drug discovery brainstorming session for Alzheimer's disease.

-   **Agents**:
    1.  `Senior Biochemist`: Researches the molecular basis of Alzheimer's.
    2.  `Gene Therapist`: Proposes a novel RNA-based therapy based on the biochemist's findings.
    3.  `The Pragmatist ("Man off the Street")`: Critiques the proposed therapy from a patient's perspective, focusing on real-world feasibility and cost.
-   **Usage**: A powerful example of combining expert knowledge with practical, human-centered feedback.

#### Code (`drug_discovery_crew.py`)

```python
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Initialize the internet search tool
search_tool = SerperDevTool()

# --- 1. Define Your Specialist Agents ---

# Agent 1: The Biochemist
biochemist = Agent(
  role='Senior Biochemist specializing in neurodegenerative diseases',
  goal='Analyze and summarize the latest biochemical research on Alzheimer\'s, focusing on protein misfolding and amyloid plaques.',
  backstory=(
    "You are a world-renowned biochemist with deep expertise in the molecular mechanisms of "
    "diseases like Alzheimer's. You are a master at reading dense scientific papers and "
    "extracting the most critical, actionable insights."
  ),
  verbose=True,
  allow_delegation=False,
  tools=[search_tool] # This agent can search the web
)

# Agent 2: The Gene Therapist
gene_therapist = Agent(
  role='Expert Geneticist and Gene Therapy Researcher',
  goal='Propose novel gene therapy or RNA-based therapeutic approaches based on biochemical findings.',
  backstory=(
    "You are at the cutting edge of genetic medicine. You think in terms of CRISPR, siRNA, and mRNA "
    "delivery vectors. You take foundational biochemical research and envision how to turn it into "
    "a programmable, information-based therapeutic."
  ),
  verbose=True,
  allow_delegation=False
  # This agent does not need to search; it synthesizes the first agent's work
)

# Agent 3: "The Man off the Street" (The Pragmatist)
pragmatist = Agent(
    role='A practical, common-sense patient advocate',
    goal='Critique proposed therapeutic approaches from the perspective of a real patient. Focus on simplicity, cost, and real-world feasibility.',
    backstory=(
        "You are not a scientist. You represent the patient. You ask the naive, "
        "common-sense questions that experts often forget. Is this treatment incredibly expensive? "
        "Does it require weekly hospital visits? Is it more complicated than it needs to be? "
        "Your job is to ground the brilliant science in human reality."
    ),
    verbose=True,
    allow_delegation=False
)


# --- 2. Create the Tasks for Your Agents ---

# Task for the Biochemist: Find the foundational research
research_task = Task(
  description=(
    "Conduct a comprehensive search on the biochemical basis of Alzheimer's Disease. "
    "Specifically, find the latest research (last 2 years) on the role of Tau proteins and Amyloid-beta plaques. "
    "Summarize the key molecular targets that are currently being investigated."
  ),
  expected_output='A concise, bullet-point summary of 3-5 key molecular targets and a brief explanation of each.',
  agent=biochemist
)

# Task for the Gene Therapist: Propose a novel solution
propose_therapy_task = Task(
  description=(
    "Based on the identified molecular targets, propose one novel RNA-based therapeutic approach. "
    "Briefly describe the mechanism of action (e.g., using siRNA to silence a specific gene, "
    "or an mRNA vaccine to trigger an immune response against plaques). Be creative and bold."
  ),
  expected_output='A 2-paragraph proposal for a single, novel RNA-based therapy, including its target and proposed mechanism.',
  agent=gene_therapist
)

# Task for the Pragmatist: Critique the proposal
critique_task = Task(
    description=(
        "Review the proposed RNA-based therapy. From a patient's perspective, identify the top 3 "
        "potential real-world challenges or questions. For example, 'How is this delivered? A pill or an injection?', "
        "'How often would I need this treatment?', 'Will this be affordable or only for the ultra-rich?'"
    ),
    expected_output='A bullet-point list of the 3 most important, common-sense questions or concerns a patient would have about the proposed therapy.',
    agent=pragmatist
)


# --- 3. Assemble the Crew and Kick Off the Mission ---

drug_discovery_crew = Crew(
  agents=[biochemist, gene_therapist, pragmatist],
  tasks=[research_task, propose_therapy_task, critique_task],
  process=Process.sequential,
  verbose=2
)

result = drug_discovery_crew.kickoff()

print("\n\n########################")
print("## Final Synthesized Report:")
print("########################\n")
print(result)
🏁 Getting Started
Prerequisites
Python 3.8+
An API key for an LLM provider (e.g., OpenAI, Cohere, etc.).
An API key for Serper (for the web search tool).
Installation
Clone the repository:
code
Sh
git clone https://github.com/your-username/crewai-examples.git
cd crewai-examples
Install the required libraries:
code
Sh
pip install crewai crewai_tools python-dotenv
Set up your environment variables:
Create a .env file in the root of the project and add your API keys:
code
Code
# .env file
OPENAI_API_KEY="your-openai-api-key"
SERPER_API_KEY="your-serper-api-key"
The code will automatically load these variables.
Running the Scripts
Simply execute the Python file you want to run from your terminal:
code
Sh
# For the tech trend analysis
python tech_research_crew.py

# For the drug discovery simulation
python drug_discovery_crew.py
💡 Customization
These examples are just the beginning. You can easily customize them or create your own crews:
Create New Agents: Dream up any role you can think of! A financial analyst, a creative writer, a software QA tester, etc.
Design New Tasks: Define any multi-step problem you want to solve.
Add More Tools: crewai_tools offers many other tools, including file read/write, website scraping, and more. You can also create your own custom tools.
License
This project is licensed under the MIT License.
