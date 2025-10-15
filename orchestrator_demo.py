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
print(result)
