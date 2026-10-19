# Sample Project

This is a very simple Python project designed to test the **Autonomous Codebase Guardian** system.

## How to test:
1. From the AgentoCode project root, configure the Groq key and download the pinned local embedding model as described in the main README.
2. Start the CLI with `python cli.py --dir ./sample_project`. Use `/status` to check knowledge generation and indexing.
3. Ask the Guardian questions like:
   - "How does the SimpleCalculator handle division by zero?"
   - "Can you add a power function to the calculator?"
   - "Refactor `main.py` to ask for user input in a loop."
