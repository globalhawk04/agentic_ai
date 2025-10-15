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
