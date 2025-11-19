# Build Your Own AI Agent: A Step-by-Step Tutorial

## Introduction

Have you ever wanted to build an AI agent that can autonomously handle file operations, analyze data, and maintain conversation context? In this comprehensive tutorial, we'll build a sophisticated agent system from the ground up using the **GAME framework** (Goals, Actions, Memory, Environment).

By the end of this tutorial, you'll have a working agent that can:
- Navigate directories and analyze CSV files
- Make intelligent tool selections based on user intent
- Handle errors gracefully
- Maintain conversation history

Let's start simple and gradually add complexity!

## Prerequisites

Before we begin, make sure you have:
- Python 3.8+
- Basic understanding of Python classes and decorators
- A API key, you can use Cohere like me (free from https://dashboard.cohere.com/), or you can use Openai or other platforms as well
- The following packages installed:

```bash
pip install cohere langchain-cohere pandas rapidfuzz python-dotenv
```

## Step 1: Project Structure

Create a new directory for your project and set up the following structure:

```
my_agent/
├── .env                    # API keys
├── main.py                 # Entry point
├── GAME.py                 # Core framework
├── language.py             # Agent communication
├── agent.py                # Main agent class
├── response_generator.py   # LLM integration
├── tool_registry.py        # Tool management
└── tools/
    ├── file_tools.py       # File operations
    └── system_tools.py     # System utilities
```

## Step 2: Environment Setup

First, create your `.env` file:

```env
COHERE_API_KEY=your_cohere_api_key_here
OPENAI_API_KEY=your_oepnai_api_key_here(if you are using openai)
```

## Step 3: Building the GAME Framework

Let's start with the foundation: Create GAME.py.
When creating GAME.py, think of it as the engine of your agent. Just like building a car engine once and using it in different cars, GAME.py gives you reusable building blocks. GAME stands for Goal, Action, Memory and Environment that are the building blocks of the AI agent's engine. You can reuse this code whenever you are building a new agent.
Now let's have a look at the following code:

📁 File: GAME.py
🔗 Source: [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/GAME.py)

This code has 4 classes for Goal, Action, Memory and Environment. Later on in your main.py, you will write down explicitly what is the goal, how to get the action and what is the memory for your agent, but the definition of the class and its attributes are set in this code.
What this does:

**Goals** give your agent direction. The frozen=True makes them immutable - once you define a goal, it can't be accidentally changed.
**Actions** wrap Python functions so your agent can call them. The ActionRegistry keeps track of all available tools.
**Memory** stores the conversation history. Each item is a dictionary representing one piece of the conversation (user input, agent response, tool result).
**Environment** safely executes actions. If something goes wrong, it catches the error instead of crashing your agent.


## Step 4: Tool Registration System

This is where the magic happens for automatically registering your tools. When you write a Python function as a tool for your agent to use - for example, a function that lists CSV files or counts how many CSV files exist in a folder - you write the Python function, but for the agent to understand this is a tool that it can use to do these tasks, you need this tool registry.

**The Problem Without Tool Registry:** Your agent is like a person in a workshop full of tools, but they're all unlabeled. The agent doesn't know what each tool does or how to use it.

**The Solution:** This code is where Python functions get registered for your agent as tools - complete with descriptions, parameters, and everything the agent needs to understand and use them effectively.

**The Magic of Decorators:** By using decorators, any time you create a tool, it automatically becomes available for the agent. You don't need to do anything manually to tell your agent "hey, this is a new tool you can use." Just add `@register_tool` above your function, and boom - your agent instantly knows about it.

**Think of it like this:**
- **Without tool registry**: Write function → manually tell agent about it → manually describe what it does → manually list parameters
- **With tool registry**: Write function → add `@register_tool` decorator → agent automatically discovers and understands it

It's the difference between having to introduce every tool individually versus having a smart assistant that automatically catalogs everything for you.
Create tool_registry.py: This is where the magic happens for automatically registering your tools. When you write a Python function as a tool for your agent to use - for example, a function that lists CSV files or counts how many CSV files exist in a folder - you write the Python function, but for the agent to understand this is a tool that it can use to do these tasks, you need this tool registry.

📁 **File:** `tool_registry.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/tool_registry.py)

**What this does**: The `@register_tool` decorator automatically:
- Extracts function signatures
- Converts Python types to JSON schema
- Organizes tools by tags
- Makes functions available to your agent

## Step 5: Your First Tool

Create `tools/system_tools.py` with a simple termination tool:

```python
# tools/system_tools.py
from tool_registry import register_tool

@register_tool(tags=["system"], terminal=True)
def terminate(message: str) -> str:
    """Terminates the agent's execution with a final message."""
    return f"{message}\nTerminating..."
```
**What this does**: This creates a simple tool that ends the agent's execution. The `terminal=True` parameter tells the agent this action should stop the conversation.

For more examples see:
📁 File: file_tools.py, system_tools
🔗 Source: [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/tools/)


## Step 6: LLM Integration

Create `response_generator.py`: This is the bridge between your agent and the language model. When your agent needs to think, plan, or respond to user input, it sends a request to the LLM through this module and gets back intelligent responses.

📁 **File:** `response_generator.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/response_generator.py)

**What this does**: This handles communication with the Cohere API, including:
- Converting your prompts to LangChain format
- Managing tool calling
- Timeout protection to prevent hanging
- Error handling for API failures

## Step 7: Agent Communication

Create `agent_language.py`. This is where we define how the agent communicates:

📁 **File:** `agent_language.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/agent_language.py)

**What this does**: 
- Converts your goals into system messages for the LLM
- Formats the conversation history appropriately
- Converts your actions into tool definitions the LLM can understand
- Parses the LLM's responses back into tool calls

## Step 8: The Agent Loop

Create `agent_loop.py` - the heart of your agent:

📁 **File:** `agent_loop.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/agent_loop.py)

**What this does**: This is your agent's "brain" It:
1. Takes user input and builds a complete prompt
2. Gets a decision from the LLM
3. Executes the chosen action safely
4. Updates its memory with the results
5. Continues until the task is complete

## Step 9: Basic File Operations

Create `tools/file_tools.py`. Let's start with a simple file reader:

```python
# tools/file_tools.py
import os
from tool_registry import register_tool

@register_tool(tags=["file_operations"])
def list_files(path: str = ".") -> str:
    """List all files in a directory."""
    try:
        if not os.path.exists(path):
            return f"❌ Path not found: {path}"
        
        files = os.listdir(path)
        if not files:
            return f"📁 Directory '{path}' is empty"
        
        return f"📁 Files in '{path}':\n" + "\n".join([f"  📄 {f}" for f in files])
    except Exception as e:
        return f"❌ Error listing files: {e}"

@register_tool(tags=["file_operations"])
def read_file(filename: str) -> str:
    """Read the content of a text file."""
    try:
        if not os.path.exists(filename):
            return f"❌ File not found: {filename}"
        
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"📄 Content of '{filename}':\n{content}"
    except Exception as e:
        return f"❌ Error reading file: {e}"
```

**What this does**: These are your first real tools! They let your agent:
- List files in directories
- Read text files
- Handle errors gracefully with helpful messages


for more examples see:

📁 **File:** `file_tools.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/tools/file_tools.py)

## Step 10: Putting It All Together

Create your `main.py`:

```python
# main.py
import os 
from dotenv import load_dotenv
from response_generator import generate_response
from GAME import Goal, Memory, Environment
from agent_language import AgentFunctionCallingActionLanguage, PythonActionRegistry
from agent_loop import Agent
import tools.file_tools
import tools.system_tools


load_dotenv()  # Load variables from .env file


api_key = os.getenv("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = api_key




# Define the agent's goals
goals = [
    Goal(priority=1,
         name="Explore Files",
         description="Navigate folders and list available CSV files."),
    Goal(priority=2,
         name="Analyze CSV Files",
         description="Count, identify, match, and inspect columns in CSV files using fuzzy matching."),
    Goal(priority=3,
         name="Clean and Consolidate Data",
         description="Clean CSV files and merge them into a consolidated report."),
    Goal(priority=4,
         name="Terminate",
         description="Call terminate when the user explicitly asks or the task is complete.")
]


# Create the agent
agent = Agent(
    goals=goals,
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=PythonActionRegistry(tags=["file_operations", "system"]),
    generate_response=generate_response,
    environment=Environment()
)


# Start interactive loop
memory = Memory()

print("Hi, I'm your agent. Ask me to explore directories, list CSVs, or analyze files. Type 'exit' to quit.\n")

while True:
    user_input = input("🧑 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    agent.set_current_task(memory, user_input)
    prompt = agent.construct_prompt(agent.goals, memory, agent.actions)
    response = agent.prompt_llm_for_action(prompt)

    action, invocation = agent.get_action(response)
    result = agent.environment.execute_action(action, invocation["args"])

    agent.update_memory(memory, response, result)

    if result.get("tool_executed"):
        print(f"\n🤖 Agent: {result['result']}\n")
    else:
        print(f"\n🤖 Agent encountered an error:\n{result['error']}\n")

    if agent.should_terminate(response):
        print("Agent has decided to terminate.")
        break
```

## Step 11: Test Your Basic Agent

In your working directory, make a new folder and call it "input_csvs" and put some csv files in that folder. 

Run your agent:

```bash
python main.py
```

Try these commands:
- "go to input_csv folders"
- "how many csv files are in input_csv folder"
- "read "input_csv/your_file1.csv""
- "exit"

Here is an example of how my AI agent interacts:
<img width="980" height="924" alt="image" src="https://github.com/user-attachments/assets/17290a9e-3ea9-4ebe-9f3c-8ba65f7b8896" />


**Congratulations!** You now have a working AI agent that can:
- Understand natural language requests
- Choose appropriate tools
- Execute file operations safely
- Maintain conversation context
- Handle errors gracefully




## Conclusion

You've built a sophisticated AI agent system from scratch. Here's what you've accomplished:
