import os
import json
import time
from openai import OpenAI  # Using OpenAI for this example, but any powerful LLM works

# --- Configuration ---
# Make sure you have your OPENAI_API_KEY set as an environment variable
client = OpenAI()

# --- The Core HITL Workflow ---

class HumanInTheLoop:
    """A simple class to manage the HITL process."""

    def get_human_validation(self, proposals: dict) -> str | None:
        """
        Presents AI-generated proposals to a human for a final decision.
        """
        print("\n" + "="*50)
        print("👤 HUMAN-IN-THE-LOOP VALIDATION REQUIRED 👤")
        print("="*50)
        print("\nThe AI has analyzed the situation and recommends the following options:")

        if not proposals or "options" not in proposals:
            print("  -> AI failed to generate valid proposals.")
            return None

        for i, option in enumerate(proposals["options"]):
            print(f"\n--- OPTION {i+1}: {option['name']} ---")
            print(f"  - Strategy: {option['strategy']}")
            print(f"  - Estimated Cost Impact: ${option['cost_impact']:,}")
            print(f"  - Estimated ETA Impact: {option['eta_impact_hours']} hours")
            print(f"  - Risk Assessment: {option['risk']}")

        print("\n" + "-"*50)
        
        while True:
            try:
                choice = input(f"Please approve an option by number (1-{len(proposals['options'])}) or type 'reject' to abort: ")
                if choice.lower() == 'reject':
                    return "REJECTED"
                
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(proposals["options"]):
                    return proposals["options"][choice_index]["name"]
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

def ai_logistics_analyst(situation: str) -> dict:
    """
    An AI agent that analyzes a logistics problem and proposes solutions.
    """
    print("\n" + "="*50)
    print("🤖 AI LOGISTICS ANALYST ACTIVATED 🤖")
    print("="*50)
    print(f"Analyzing situation: {situation}")
    
    # In a real app, this would be a more complex system prompt
    system_prompt = (
        "You are an expert logistics analyst. Your job is to analyze a shipping disruption "
        "and propose three distinct, actionable solutions. For each solution, you must provide a name, a strategy, "
        "an estimated cost impact, an ETA impact in hours, and a brief risk assessment. "
        "Your entire response MUST be a single, valid JSON object with a key 'options' containing a list of these three solutions."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": situation}
            ]
        )
        proposals = json.loads(response.choices[0].message.content)
        print("  -> AI has generated three viable proposals.")
        return proposals
    except Exception as e:
        print(f"  -> ERROR: AI analysis failed: {e}")
        return {"options": []}

def execute_final_plan(approved_plan: str):
    """
    Simulates the execution of the human-approved plan.
    """
    print("\n" + "="*50)
    print("✅ EXECUTION CONFIRMED ✅")
    print("="*50)
    print(f"Executing the human-approved plan: '{approved_plan}'")
    print("  -> Rerouting instructions dispatched to driver.")
    print("  -> Notifying customer of potential delay.")
    print("  -> Updating logistics database with new ETA.")
    print("\nWorkflow complete.")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. The problem arises
    current_situation = (
        "Critical shipment #734-A, en route from Los Angeles to New York, is currently in Kansas. "
        "A severe weather alert has been issued for a massive storm system directly in its path, "
        "projected to cause closures on I-70 and I-80 for the next 48 hours. The current ETA is compromised."
    )

    # 2. The AI does the heavy lifting
    ai_proposals = ai_logistics_analyst(current_situation)

    # 3. The Human is brought "in the loop" for the critical decision
    hitl_validator = HumanInTheLoop()
    final_decision = hitl_validator.get_human_validation(ai_proposals)

    # 4. The system executes based on the human's choice
    if final_decision and final_decision != "REJECTED":
        execute_final_plan(final_decision)
    else:
        print("\n" + "="*50)
        print("❌ EXECUTION ABORTED ❌")
        print("="*50)
        print("Human operator rejected all proposals. No action will be taken.")
