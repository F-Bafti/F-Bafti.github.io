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
- A Cohere API key (free from https://dashboard.cohere.com/)
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
├── agent_language.py       # Agent communication
├── agent_loop.py          # Main agent logic
├── response_generator.py   # LLM integration
├── tool_registry.py       # Tool management
└── tools/
    ├── file_tools.py      # File operations
    └── system_tools.py    # System utilities
```

## Step 2: Environment Setup

First, create your `.env` file:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

## Step 3: Building the GAME Framework

Let's start with the foundation: Create `GAME.py`.
When creating GAME.py, think of it as the engine of your agent. Just like building a car engine once and using it in different cars, GAME.py gives you reusable building blocks. GAME stands for Goal, Action, Memory and Environment that are the building blocks of the AI agent's engine.

📁 **File:** `GAME.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/GAME.py)

**What this does**: 
- **Goals** give your agent direction. The `frozen=True` makes them immutable - once you define a goal, it can't be accidentally changed.
- **Actions** wrap Python functions so your agent can call them. The ActionRegistry keeps track of all available tools.
- **Memory** stores the conversation history. Each item is a dictionary representing one piece of the conversation (user input, agent response, tool result).
- **Environment** safely executes actions. If something goes wrong, it catches the error instead of crashing your agent.

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

## Step 6: LLM Integration

Create `response_generator.py`: This is the bridge between your agent and the language model. When your agent needs to think, plan, or respond to user input, it sends a request to the LLM through this module and gets back intelligent responses.

📁 **File:** `tool_registry.py`  
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

```python
# agent_loop.py
import json
from typing import List, Callable, Any
from GAME import Goal, Environment, ActionRegistry, Memory
from agent_language import AgentLanguage
from response_generator import Prompt

class Agent:
    def __init__(self, goals: List[Goal], agent_language: AgentLanguage,
                 action_registry: ActionRegistry, generate_response: Callable[[Prompt], str],
                 environment: Environment):
        """Initialize an agent with its core GAME components."""
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment

    def construct_prompt(self, goals: List[Goal], memory: Memory, actions: ActionRegistry) -> Prompt:
        """Build prompt with memory context."""
        return self.agent_language.construct_prompt(
            actions=actions.get_actions(),
            environment=self.environment,
            goals=goals,
            memory=memory
        )

    def get_action(self, response):
        """Parse response and get the corresponding action."""
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        return action, invocation

    def should_terminate(self, response: str) -> bool:
        """Check if the agent should stop executing."""
        action_def, _ = self.get_action(response)
        return action_def.terminal if action_def else False

    def set_current_task(self, memory: Memory, task: str):
        """Add user input to memory."""
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self, memory: Memory, response: str, result: dict):
        """Update memory with agent's decision and environment's response."""
        new_memories = [
            {"type": "assistant", "content": response},
            {"type": "environment", "content": json.dumps(result)}
        ]
        for m in new_memories:
            memory.add_memory(m)

    def prompt_llm_for_action(self, full_prompt: Prompt) -> str:
        """Get response from LLM."""
        return self.generate_response(full_prompt)

    def run(self, user_input: str, memory=None, max_iterations: int = 50) -> Memory:
        """Execute the main agent loop."""
        memory = memory or Memory()
        self.set_current_task(memory, user_input)

        for _ in range(max_iterations):
            # 1. Build prompt with current context
            prompt = self.construct_prompt(self.goals, memory, self.actions)
            
            # 2. Get agent's decision
            response = self.prompt_llm_for_action(prompt)
            print(f"Agent Decision: {response}")

            # 3. Execute the chosen action
            action, invocation = self.get_action(response)
            result = self.environment.execute_action(action, invocation["args"])
            print(f"Action Result: {result}")

            # 4. Update memory with what happened
            self.update_memory(memory, response, result)

            # 5. Check if we should stop
            if self.should_terminate(response):
                break

        return memory
```

**What this does**: This is your agent's "brain." It:
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

# Load environment variables
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = api_key

# Define your agent's goals
goals = [
    Goal(priority=1, name="File Operations", 
         description="Help users list files and read file contents."),
    Goal(priority=2, name="Terminate", 
         description="Call terminate when the user asks to quit or task is complete.")
]

# Create the agent
agent = Agent(
    goals=goals,
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=PythonActionRegistry(tags=["file_operations", "system"]),
    generate_response=generate_response,
    environment=Environment()
)

# Interactive loop
memory = Memory()
print("🤖 Hi! I can help you with file operations. Ask me to list files or read files!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("🧑 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Set the task and get a prompt
    agent.set_current_task(memory, user_input)
    prompt = agent.construct_prompt(agent.goals, memory, agent.actions)
    
    # Get agent's decision
    response = agent.prompt_llm_for_action(prompt)
    
    # Execute the action
    action, invocation = agent.get_action(response)
    result = agent.environment.execute_action(action, invocation["args"])
    
    # Update memory and show result
    agent.update_memory(memory, response, result)
    
    if result.get("tool_executed"):
        print(f"\n🤖 Agent: {result['result']}\n")
    else:
        print(f"\n🤖 Agent error: {result['error']}\n")
    
    # Check for termination
    if agent.should_terminate(response):
        print("🤖 Agent: Goodbye!")
        break
```

## Step 11: Test Your Basic Agent

Run your agent:

```bash
python main.py
```

Try these commands:
- "list files in the current directory"
- "read the main.py file"
- "what files are here?"
- "exit"

**Congratulations!** You now have a working AI agent that can:
- Understand natural language requests
- Choose appropriate tools
- Execute file operations safely
- Maintain conversation context
- Handle errors gracefully

## Step 12: Adding Advanced Features

Now let's add some advanced CSV analysis capabilities. Update `tools/file_tools.py`:

```python
# Add to tools/file_tools.py
import pandas as pd
from rapidfuzz import process, fuzz

# Expected columns for CSV analysis
EXPECTED_COLUMNS = [
    "Name", "Age", "Email", "Department", "Salary"
]

@register_tool(tags=["file_operations", "csv"])
def list_csv_files(path: str = ".") -> str:
    """List only CSV files in a directory."""
    try:
        if not os.path.exists(path):
            return f"❌ Path not found: {path}"
        
        files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
        if not files:
            return f"📁 No CSV files found in '{path}'"
        
        return f"📊 CSV files in '{path}':\n" + "\n".join([f"  📊 {f}" for f in files])
    except Exception as e:
        return f"❌ Error listing CSV files: {e}"

@register_tool(tags=["file_operations", "csv"])
def analyze_csv_columns(file_path: str) -> str:
    """Analyze columns in a CSV file and match them to expected columns."""
    try:
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"
        
        df = pd.read_csv(file_path)
        if df.empty:
            return f"❌ CSV file '{file_path}' is empty"
        
        actual_columns = [col.strip() for col in df.columns]
        results = []
        
        results.append(f"📊 Analysis of '{file_path}':")
        results.append(f"📏 Rows: {len(df)}, Columns: {len(actual_columns)}")
        results.append("\n🔍 Column Matching:")
        
        for expected in EXPECTED_COLUMNS:
            if expected in actual_columns:
                results.append(f"  ✅ {expected} → {expected} (exact match)")
            else:
                # Try fuzzy matching
                match, score, _ = process.extractOne(expected, actual_columns, scorer=fuzz.token_set_ratio)
                if score >= 70:
                    results.append(f"  🔶 {expected} → {match} ({score:.1f}% match)")
                else:
                    results.append(f"  ❌ {expected} → Not found")
        
        results.append(f"\n📋 Actual columns: {', '.join(actual_columns)}")
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Error analyzing CSV: {e}"

@register_tool(tags=["file_operations", "csv"])
def count_csv_files(path: str = ".") -> str:
    """Count CSV files in a directory."""
    try:
        if not os.path.exists(path):
            return f"❌ Path not found: {path}"
        
        csv_files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
        count = len(csv_files)
        
        return f"📊 Found {count} CSV file{'s' if count != 1 else ''} in '{path}'"
    except Exception as e:
        return f"❌ Error counting CSV files: {e}"
```

Update your goals in `main.py`:

```python
goals = [
    Goal(priority=1, name="File Operations", 
         description="Help users navigate directories and list files."),
    Goal(priority=2, name="CSV Analysis", 
         description="Analyze CSV files, count them, and match column names."),
    Goal(priority=3, name="Terminate", 
         description="Call terminate when the user asks to quit or task is complete.")
]
```

And update the action registry to include CSV tools:

```python
action_registry=PythonActionRegistry(tags=["file_operations", "csv", "system"]),
```

## Step 13: Test Advanced Features

Create a test CSV file called `employees.csv`:

```csv
Full Name,Years,Email Address,Dept,Monthly Pay
John Doe,25,john@company.com,Engineering,5000
Jane Smith,30,jane@company.com,Marketing,4500
```

Now test your enhanced agent:
- "list csv files"
- "analyze the employees.csv file"  
- "count csv files in this directory"
- "how many csv files are here?"

## Step 14: Adding Intelligence with Better Tool Selection

Let's make your agent smarter about choosing tools. Update the `format_goals` method in `agent_language.py`:

```python
def format_goals(self, goals: List[Goal]) -> List:
    """Convert goals into system messages with tool selection guidance."""
    goal_text = "\n\n".join([f"{goal.name}:\n{goal.description}" for goal in goals])
    
    tool_guidance = """
🎯 TOOL SELECTION GUIDE:
- "list files" or "what files are here" → list_files()
- "list csv files" or "show csv files" → list_csv_files()  
- "count csv files" or "how many csv files" → count_csv_files()
- "analyze [filename]" or "check columns in [filename]" → analyze_csv_columns(file_path="[filename]")
- "read [filename]" → read_file(filename="[filename]")
- "quit", "exit", or "done" → terminate()

Always choose the most specific tool for the user's request.
    """
    
    return [{"role": "system", "content": goal_text + "\n" + tool_guidance}]
```

## Step 15: Error Handling and Robustness

Let's add better error handling. Create a custom exception handler in `GAME.py`:

```python
# Add to GAME.py
import logging

class SafeEnvironment(Environment):
    def __init__(self):
        super().__init__()
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action with enhanced error handling."""
        self.logger.info(f"Executing action: {action.name} with args: {args}")
        
        try:
            # Validate arguments
            if not isinstance(args, dict):
                raise ValueError("Arguments must be a dictionary")
            
            result = action.execute(**args)
            self.logger.info(f"Action {action.name} completed successfully")
            return self.format_result(result)
            
        except Exception as e:
            self.logger.error(f"Action {action.name} failed: {str(e)}")
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "action_name": action.name,
                "action_args": args
            }
```

Update your main.py to use the enhanced environment:

```python
environment=SafeEnvironment()
```

## Step 16: Multi-Language Support (Advanced)

For handling Persian/Farsi content, add these constants to `tools/file_tools.py`:

```python
# Persian column mappings
FARSI_SYNONYM_MAP = {
    "اسم مرکز": "نام مرکز",
    "نام مرکز آموزشی": "نام مرکز", 
    "نام و نام خانوادگی معلم": "نام معلم",
    "نام مدرس": "نام معلم",
    "تعداد دانش آموزان": "تعداد دانش آموزان شرکت کننده"
}

@register_tool(tags=["file_operations", "csv", "multilingual"])
def analyze_farsi_csv(file_path: str, cutoff: float = 70) -> str:
    """Analyze a Farsi/Persian CSV file with cultural context."""
    try:
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"
        
        df = pd.read_csv(file_path, encoding='utf-8')
        if df.empty:
            return f"❌ CSV file '{file_path}' is empty"
        
        actual_columns = [col.strip() for col in df.columns]
        results = []
        
        results.append(f"📊 تحلیل فایل '{file_path}':")
        results.append(f"📏 ردیف‌ها: {len(df)}, ستون‌ها: {len(actual_columns)}")
        results.append("\n🔍 تطبیق ستون‌ها:")
        
        for expected in FARSI_EXPECTED_COLUMNS:
            # Check for exact match
            if expected in actual_columns:
                results.append(f"  ✅ {expected} → {expected} (تطبیق دقیق)")
                continue
            
            # Check synonyms
            synonym_match = None
            for actual in actual_columns:
                if FARSI_SYNONYM_MAP.get(actual) == expected:
                    synonym_match = actual
                    break
            
            if synonym_match:
                results.append(f"  🔶 {expected} → {synonym_match} (مترادف)")
                continue
            
            # Fuzzy matching
            match, score, _ = process.extractOne(expected, actual_columns, scorer=fuzz.token_set_ratio)
            if score >= cutoff:
                results.append(f"  🔶 {expected} → {match} ({score:.1f}% تطبیق)")
            else:
                results.append(f"  ❌ {expected} → یافت نشد")
        
        results.append(f"\n📋 ستون‌های موجود: {', '.join(actual_columns)}")
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ خطا در تحلیل CSV: {e}"
```

## Step 17: Batch Processing Tools

Add batch processing capabilities:

```python
# Add to tools/file_tools.py

@register_tool(tags=["file_operations", "batch"])
def process_all_csv_files(path: str = ".") -> str:
    """Process all CSV files in a directory and provide summary."""
    try:
        if not os.path.exists(path):
            return f"❌ Path not found: {path}"
        
        csv_files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
        if not csv_files:
            return f"📁 No CSV files found in '{path}'"
        
        results = [f"📊 Processing {len(csv_files)} CSV files in '{path}':\n"]
        
        total_rows = 0
        for csv_file in csv_files:
            try:
                file_path = os.path.join(path, csv_file)
                df = pd.read_csv(file_path)
                rows = len(df)
                cols = len(df.columns)
                total_rows += rows
                
                results.append(f"  📄 {csv_file}: {rows} rows, {cols} columns")
            except Exception as e:
                results.append(f"  ❌ {csv_file}: Error - {str(e)}")
        
        results.append(f"\n📈 Summary: {total_rows} total rows across {len(csv_files)} files")
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Error processing CSV files: {e}"

@register_tool(tags=["file_operations", "search"])
def find_files_by_keyword(keyword: str, path: str = ".") -> str:
    """Find files containing a specific keyword in their name."""
    try:
        if not os.path.exists(path):
            return f"❌ Path not found: {path}"
        
        all_files = os.listdir(path)
        matching_files = [f for f in all_files if keyword.lower() in f.lower()]
        
        if not matching_files:
            return f"🔍 No files found containing '{keyword}' in '{path}'"
        
        results = [f"🔍 Files containing '{keyword}' in '{path}':\n"]
        results.extend([f"  📄 {f}" for f in matching_files])
        
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Error searching files: {e}"
```

## Step 18: Memory Management Improvements

Enhance your memory system to handle large conversations. Update `GAME.py`:

```python
class AdvancedMemory(Memory):
    def __init__(self, max_items: int = 100):
        super().__init__()
        self.max_items = max_items
    
    def add_memory(self, memory: dict):
        """Add memory with automatic cleanup."""
        super().add_memory(memory)
        
        # Keep only recent memories to prevent context overflow
        if len(self.items) > self.max_items:
            # Keep system messages and recent items
            system_messages = [m for m in self.items if m.get("type") == "system"]
            recent_messages = [m for m in self.items if m.get("type") != "system"][-self.max_items + len(system_messages):]
            self.items = system_messages + recent_messages
    
    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        user_messages = [m for m in self.items if m.get("type") == "user"]
        tool_executions = [m for m in self.items if m.get("type") == "environment"]
        
        return f"📊 Conversation Summary:\n  💬 User messages: {len(user_messages)}\n  🔧 Tool executions: {len(tool_executions)}"

    def search_memory(self, keyword: str) -> List[Dict]:
        """Search memory for specific keywords."""
        matching_items = []
        for item in self.items:
            content = str(item.get("content", "")).lower()
            if keyword.lower() in content:
                matching_items.append(item)
        return matching_items
```

## Step 19: Enhanced Agent with Conversation Management

Update your agent to use advanced features. Create a new enhanced main file `enhanced_main.py`:

```python
# enhanced_main.py
import os
from dotenv import load_dotenv
from response_generator import generate_response
from GAME import Goal, AdvancedMemory, SafeEnvironment
from agent_language import AgentFunctionCallingActionLanguage, PythonActionRegistry
from agent_loop import Agent
import tools.file_tools
import tools.system_tools

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = api_key

# Enhanced goals with more capabilities
goals = [
    Goal(priority=1, name="File Navigation", 
         description="Navigate directories, list files, and search for specific files."),
    Goal(priority=2, name="CSV Analysis", 
         description="Analyze CSV files, match columns, and handle both English and Farsi content."),
    Goal(priority=3, name="Batch Processing", 
         description="Process multiple files at once and provide summaries."),
    Goal(priority=4, name="Conversation Management", 
         description="Maintain context and help users with follow-up questions."),
    Goal(priority=5, name="Terminate", 
         description="End the conversation when requested or task is complete.")
]

# Create enhanced agent
agent = Agent(
    goals=goals,
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=PythonActionRegistry(tags=["file_operations", "csv", "batch", "search", "multilingual", "system"]),
    generate_response=generate_response,
    environment=SafeEnvironment()
)

# Enhanced interactive loop
memory = AdvancedMemory(max_items=50)  # Limit memory to prevent context overflow

print("🤖 Enhanced CSV Analysis Agent")
print("=" * 40)
print("I can help you with:")
print("  📁 File operations (list, read, search)")
print("  📊 CSV analysis (English and Farsi)")
print("  🔄 Batch processing")
print("  🔍 Intelligent tool selection")
print("\nType 'help' for examples or 'exit' to quit.\n")

while True:
    user_input = input("🧑 You: ").strip()
    
    if user_input.lower() in ["exit", "quit"]:
        break
    
    if user_input.lower() == "help":
        print("\n🔧 Example commands:")
        print("  • 'list csv files'")
        print("  • 'analyze employees.csv'")
        print("  • 'count all csv files'")
        print("  • 'process all csv files'")
        print("  • 'find files with sales in the name'")
        print("  • 'analyze farsi csv data.csv'")
        print("  • 'show conversation summary'")
        print()
        continue
    
    if user_input.lower() == "show conversation summary":
        print(f"\n🤖 {memory.get_summary()}\n")
        continue
    
    # Process the request
    agent.set_current_task(memory, user_input)
    prompt = agent.construct_prompt(agent.goals, memory, agent.actions)
    response = agent.prompt_llm_for_action(prompt)
    action, invocation = agent.get_action(response)
    result = agent.environment.execute_action(action, invocation["args"])
    agent.update_memory(memory, response, result)
    
    # Display result
    if result.get("tool_executed"):
        print(f"\n🤖 Agent: {result['result']}\n")
    else:
        print(f"\n🤖 Agent error: {result['error']}\n")
    
    # Check for termination
    if agent.should_terminate(response):
        print("🤖 Agent: Task completed. Goodbye!")
        break
```

## Step 20: Testing Your Complete System

Create some test files to demonstrate your agent's capabilities:

**test_data/employees.csv:**
```csv
Full Name,Years,Email Address,Dept,Monthly Pay
John Doe,25,john@company.com,Engineering,5000
Jane Smith,30,jane@company.com,Marketing,4500
Bob Johnson,28,bob@company.com,Sales,4800
```

**test_data/courses_farsi.csv:**
```csv
نام مرکز,نوع دوره,نام معلم,تعداد دانش آموزان
مرکز تهران,ریاضی,احمد احمدی,25
مرکز اصفهان,فیزیک,فاطمه فاطمی,30
```

**test_data/sales.csv:**
```csv
Product,Quantity,Price,Date
Laptop,10,1200,2024-01-15
Mouse,50,25,2024-01-16
```

Now test with these commands:
```bash
python enhanced_main.py
```

Try these examples:
- "list csv files in test_data"
- "analyze test_data/employees.csv"
- "process all csv files in test_data"
- "find files with sales in the name"
- "analyze farsi csv test_data/courses_farsi.csv"
- "count csv files in test_data"
- "show conversation summary"

## Step 21: Deployment and Production Considerations

### Error Monitoring
Add comprehensive logging:

```python
# Create logs/agent_logger.py
import logging
import os
from datetime import datetime

def setup_logger(name: str, log_file: str = None):
    """Set up a logger with both file and console output."""
    if not log_file:
        log_file = f"logs/agent_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
```

### Configuration Management
Create `config.py`:

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class AgentConfig:
    # LLM Settings
    model_name: str = "command-r-plus"
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout: int = 30
    
    # Agent Settings
    max_iterations: int = 50
    memory_limit: int = 100
    
    # File Settings
    supported_file_types: list = None
    max_file_size_mb: int = 10
    
    # Fuzzy Matching
    fuzzy_cutoff: float = 70.0
    
    def __post_init__(self):
        if self.supported_file_types is None:
            self.supported_file_types = ['.csv', '.txt', '.json']
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "command-r-plus"),
            max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            timeout=int(os.getenv("TIMEOUT", "30")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "50")),
            memory_limit=int(os.getenv("MEMORY_LIMIT", "100")),
            fuzzy_cutoff=float(os.getenv("FUZZY_CUTOFF", "70.0"))
        )
```

### Production-Ready Main File
Create `production_main.py`:

```python
# production_main.py
import os
import sys
from dotenv import load_dotenv
from config import AgentConfig
from logs.agent_logger import setup_logger
from response_generator import generate_response
from GAME import Goal, AdvancedMemory, SafeEnvironment
from agent_language import AgentFunctionCallingActionLanguage, PythonActionRegistry
from agent_loop import Agent
import tools.file_tools
import tools.system_tools

def main():
    # Setup
    load_dotenv()
    config = AgentConfig.from_env()
    logger = setup_logger("csv_agent")
    
    # Validate API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.error("COHERE_API_KEY not found in environment variables")
        sys.exit(1)
    
    os.environ["COHERE_API_KEY"] = api_key
    logger.info("Starting CSV Analysis Agent")
    
    # Goals
    goals = [
        Goal(priority=1, name="File Navigation", 
             description="Navigate directories, list files, and search for specific files."),
        Goal(priority=2, name="CSV Analysis", 
             description="Analyze CSV files with intelligent column matching for multiple languages."),
        Goal(priority=3, name="Batch Processing", 
             description="Process multiple files efficiently with progress reporting."),
        Goal(priority=4, name="Data Quality", 
             description="Validate data quality and suggest improvements."),
        Goal(priority=5, name="Terminate", 
             description="End conversation when task is complete or user requests.")
    ]
    
    # Create agent
    try:
        agent = Agent(
            goals=goals,
            agent_language=AgentFunctionCallingActionLanguage(),
            action_registry=PythonActionRegistry(tags=["file_operations", "csv", "batch", "search", "multilingual", "system"]),
            generate_response=generate_response,
            environment=SafeEnvironment()
        )
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Interactive session
    memory = AdvancedMemory(max_items=config.memory_limit)
    
    print("🤖 CSV Analysis Agent v2.0")
    print("=" * 50)
    print("🚀 Production-ready agent for file analysis")
    print("📊 Supports English and Persian/Farsi content")
    print("🔧 Type 'help' for commands or 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("🧑 You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested exit")
                break
            
            if user_input.lower() == "help":
                show_help()
                continue
            
            if user_input.lower() in ["status", "summary"]:
                print(f"\n🤖 {memory.get_summary()}")
                print(f"🔧 Max iterations: {config.max_iterations}")
                print(f"💾 Memory limit: {config.memory_limit}\n")
                continue
            
            # Process request
            logger.info(f"Processing user request: {user_input}")
            
            agent.set_current_task(memory, user_input)
            prompt = agent.construct_prompt(agent.goals, memory, agent.actions)
            response = agent.prompt_llm_for_action(prompt)
            action, invocation = agent.get_action(response)
            result = agent.environment.execute_action(action, invocation["args"])
            agent.update_memory(memory, response, result)
            
            # Display result
            if result.get("tool_executed"):
                print(f"\n🤖 Agent: {result['result']}\n")
                logger.info(f"Successfully executed: {action.name if action else 'unknown'}")
            else:
                print(f"\n🤖 Agent error: {result['error']}\n")
                logger.error(f"Tool execution failed: {result['error']}")
            
            # Check termination
            if agent.should_terminate(response):
                print("🤖 Agent: Task completed successfully!")
                logger.info("Agent terminated normally")
                break
                
        except KeyboardInterrupt:
            print("\n\n🤖 Agent: Interrupted by user. Goodbye!")
            logger.info("Agent interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            logger.error(f"Unexpected error: {e}")
            print("🤖 Agent: Continuing...\n")

def show_help():
    """Display help information."""
    print("\n🔧 Available Commands:")
    print("=" * 30)
    print("📁 File Operations:")
    print("  • 'list files' - Show all files")
    print("  • 'list csv files' - Show only CSV files")
    print("  • 'read filename.txt' - Read a text file")
    print()
    print("📊 CSV Analysis:")
    print("  • 'analyze data.csv' - Analyze CSV structure")
    print("  • 'analyze farsi csv file.csv' - Persian analysis")
    print("  • 'count csv files' - Count CSV files")
    print()
    print("🔄 Batch Operations:")
    print("  • 'process all csv files' - Batch analyze all CSVs")
    print("  • 'find files with keyword' - Search by filename")
    print()
    print("ℹ️  System:")
    print("  • 'status' - Show system status")
    print("  • 'help' - Show this help")
    print("  • 'exit' - Quit the agent")
    print()

if __name__ == "__main__":
    main()
```

## Conclusion

Congratulations! You've built a sophisticated AI agent system from scratch. Here's what you've accomplished:

### 🎯 Core Achievements
1. **GAME Framework**: Created a modular architecture with Goals, Actions, Memory, and Environment
2. **Tool System**: Built a decorator-based tool registration system
3. **LLM Integration**: Connected to Cohere's API with robust error handling
4. **File Operations**: Implemented intelligent file analysis tools
5. **Multi-language Support**: Added Persian/Farsi content handling
6. **Production Features**: Added logging, configuration, and error monitoring

### 🚀 Key Features Your Agent Has
- **Intelligent Tool Selection**: Chooses the right tool based on user intent
- **Context Awareness**: Maintains conversation history and context
- **Error Recovery**: Gracefully handles failures without crashing
- **Fuzzy Matching**: Handles column name variations and synonyms
- **Batch Processing**: Efficiently processes multiple files
- **Multi-language Support**: Works with both English and Persian content
- **Memory Management**: Prevents context overflow in long conversations
- **Production Ready**: Includes logging, configuration, and monitoring

### 🛠️ Architecture Benefits
- **Modular**: Easy to add new tools and capabilities
- **Extensible**: Simple to adapt for new domains and use cases
- **Maintainable**: Clear separation of concerns
- **Testable**: Each component can be tested independently
- **Scalable**: Can handle complex multi-step tasks

### 🎓 What You've Learned
- How to design agent architectures
- LLM integration patterns
- Tool calling and function selection
- Memory and context management
- Error handling in AI systems
- Multi-language content processing
- Production deployment considerations

### 🔮 Next Steps
Now that you have a solid foundation, you can extend your agent with:
- **Database Integration**: Connect to SQL databases
- **API Integrations**: Call external services
- **Data Visualization**: Generate charts and graphs
- **Web Interface**: Build a web UI for your agent
- **Deployment**: Deploy to cloud platforms
- **Advanced NLP**: Add entity extraction and sentiment analysis

Your agent is now ready to handle real-world CSV analysis tasks and can serve as a foundation for building more specialized AI systems. The modular architecture makes it easy to adapt for different domains - whether you're processing financial data, educational records, or any other structured content.

Remember: the key to building great AI agents is starting simple, testing thoroughly, and gradually adding complexity. You've successfully followed this approach and built something genuinely useful!SI_EXPECTED_COLUMNS = [
    "نام مرکز",      # Center name
    "نوع دوره",      # Course type  
    "نام معلم",      # Teacher name
    "تعداد دانش آموزان"  # Number of students
]

FAR
