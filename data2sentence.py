import json

def server_log_to_sentence(log_entry: dict) -> str:
    """
    Translates a structured server log dictionary into a human-readable "log sentence".
    The "grammar" of our sentence is a fixed order of importance:
    status -> method -> path -> latency -> user_agent
    """
    
    # Define the order of importance for our "grammar"
    grammar_order = ['status', 'method', 'path', 'latency_ms', 'user_agent']
    
    sentence_parts = []
    
    for key in grammar_order:
        value = log_entry.get(key)
        if value is not None:
            # We don't just append the value; we give it a semantic prefix
            # This helps the LLM understand the meaning of each part.
            sentence_parts.append(f"{key.upper()}_{value}")
            
    return " ".join(sentence_parts)

def create_multimodal_prompt(log_sentence: str, human_context: str) -> str:
    """
    Combines the machine-generated "log sentence" with human-provided context
    to create a rich, multimodal prompt for an LLM.
    """
    prompt = f"""
    Analyze the following server request.

    **Human Context:** "{human_context}"

    **Log Sentence:** "{log_sentence}"

    Based on both the human context and the log sentence, what is the likely user intent and should we be concerned?
    """
    return prompt

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Our raw, structured data (e.g., from a database or log file)
    raw_log = {
        "timestamp": "2025-10-26T10:00:05Z",
        "method": "GET",
        "path": "/api/v1/user/settings",
        "status": 403,
        "latency_ms": 150,
        "user_agent": "Python-requests/2.25.1"
    }

    # 2. Translate the data into the new "language"
    log_sentence = server_log_to_sentence(raw_log)
    
    print("--- Original Structured Data ---")
    print(json.dumps(raw_log, indent=2))
    
    print("\n--- Translated 'Log Sentence' ---")
    print(log_sentence)
    
    # 3. Combine with human context for a multimodal prompt
    human_context = "We've been seeing a series of failed API calls from a script, not a browser."
    final_prompt = create_multimodal_prompt(log_sentence, human_context)
    
    print("\n--- Final Multimodal Prompt for LLM ---")
    print(final_prompt)
    
    # Now, this final_prompt can be sent to any standard LLM for deep analysis.
    # The LLM can now reason about both the structured log data (as a sentence)
    # and the unstructured human observation, simultaneously.
