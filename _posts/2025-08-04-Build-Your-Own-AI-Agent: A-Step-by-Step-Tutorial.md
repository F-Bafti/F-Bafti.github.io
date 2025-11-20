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

```bash
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

```bash
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

**1. Goals** give your agent direction. The frozen=True makes them immutable - once you define a goal, it can't be accidentally changed.<br>
**2. Actions** wrap Python functions so your agent can call them. The ActionRegistry keeps track of all available tools.<br>
**3. Memory** stores the conversation history. Each item is a dictionary representing one piece of the conversation (user input, agent response, tool result).<br>
**4. Environment** safely executes actions. If something goes wrong, it catches the error instead of crashing your agent.<br>

In the **Goal** class we have the following: 

```python
@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str
```
dataclass is a class from dataclasses and allows us to define classes and make them frozen. It is ideal for Goal becasue the goal of the agent should not change during the execution of tasks.  In the Goal class we define each goal, their proirity, name and description. For example since we are building an AI agent here that should read CSV files, find their headers and cleanup and consolidate them, later in the main.py we have the following lines when we use Goal class.

```python
from GAME import Goal

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
         description="Call terminate when the user explicitly asks or the task is go to plete.")
]
```

The next class is **Action**. It creates a wrapper around a python function to make it available to the AI agent.  It has name, function, description, parameters, terminal attributes. 

```python
class Action:
    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 parameters: Dict,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        """Execute the action's function"""
        return self.function(**args)
```

through the "execute", the agent actually execute the action. And then we have an action registery which is a container to storr all available actions:

```python
class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    # Looks up an action by its name
    def get_action(self, name: str) -> Action | None:
        return self.actions.get(name, None)

    # Returns ALL actions as a list
    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())
```

What action registry does is looking up an action by its name and finding all the action availables. 
In the **Memory** class, we create another container to store the conversation history with the agent. 

```python
class Memory:
    def __init__(self):
        self.items = []  # Basic conversation history

    # Adds a new memory item to the end of the list
    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    # Returns the stored memories as a list
    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        return self.items[:limit]

    # Creates a new Memory object with system messages filtered out
    def copy_without_system_memories(self):
        """Return a copy of the memory without system memories"""
        filtered_items = [m for m in self.items if m["type"] != "system"]
        memory = Memory()
        memory.items = filtered_items
        return memory
```

Finally in **Environment** class, is where the **Action** actually get executed. It has two important sections, in "try", it attempts to run the action. if it works, then it takes the results and format it nicely for the user but if it fails to excute the function, the agent should catch the error and returns the error information instead of crashing.

```python
class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }
```

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

Create `tools/file_tools.py` with a simple termination tool:

```python
@register_tool(tags=["file_operations", "list"])
def list_csv_files() -> List[str]:
    """
    USAGE: Use when user asks 'list csv files', 'show csvs', or 'what csvs are here' while already in the target directory.
    
    Lists all CSV files in the current working directory only.
    """
    return sorted([file for file in os.listdir(".") if file.lower().endswith(".csv")])  
```

**What this does**: This creates a simple tool that list all CSV files in the current working directory only.

For more examples see:
📁 File: file_tools.py, system_tools.py
🔗 Source: [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/tools/)


## Step 6: LLM Integration

Create `response_generator.py`: This is the bridge between your agent and the language model. When your agent needs to think, plan, or respond to user input, it sends a request to the LLM through this module and gets back intelligent responses.

📁 **File:** `response_generator.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/response_generator.py)

**What this does**: This handles communication with the API, including:
- Converting your prompts to LangChain format
- Managing tool calling
- Timeout protection to prevent hanging
- Error handling for API failures

## Step 7: Agent Communication

Create `language.py`. This is where we define how the agent communicates:

📁 **File:** `language.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/language.py)

**What this does**: 
- Converts your goals into system messages for the LLM
- Formats the conversation history appropriately
- Converts your actions into tool definitions the LLM can understand
- Parses the LLM's responses back into tool calls

## Step 8: The Agent Loop

Create `agent.py` - the heart of your agent:

📁 **File:** `agent.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/AI_Agent_csv_consolidator/blob/main/agent.py)

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

```bash
===========================================================================

Hi, I'm your agent. Ask me to explore directories, list CSVs, or analyze files. Type 'exit' to quit.

🧑 You: Hi, how are you

🤖 Agent: Hello! I'm an AI assistant, so I don't have feelings, but I'm here and ready to help you with any tasks related to exploring files, analyzing CSV files, cleaning and consolidating data, or anything else you need. How can I assist you today?

🧑 You: Which folder are you in right now

🤖 Agent: /Users/fahimeh/Projects/csv_consolidation

🧑 You: can you list all the files there

🤖 Agent: Type | Size (Bytes) | Modified Date/Time      | Name
------------------------------------------------------------
DIR  |              | 2025-07-29 15:10:21 | tools
FILE | 8196         | 2025-10-15 15:58:24 | .DS_Store
FILE | 8828         | 2025-07-31 11:49:19 | two_layer_registry.ipynb
FILE | 4243         | 2025-10-15 15:39:38 | response_generator.py
DIR  |              | 2025-10-15 16:12:42 | input_csvs
FILE | 4011         | 2025-11-19 12:47:55 | GAME.py
FILE | 4858         | 2025-11-19 13:44:35 | tool_registry.py
DIR  |              | 2025-11-19 16:06:35 | __pycache__
FILE | 528          | 2025-08-04 15:16:35 | README.md
FILE | 16           | 2025-07-30 15:07:20 | .gitignore
FILE | 60           | 2025-07-29 12:18:41 | .env
FILE | 9049         | 2025-11-19 16:06:21 | language.py
FILE | 3364         | 2025-11-19 14:19:41 | agent.py
DIR  |              | 2025-11-11 11:16:12 | .git
FILE | 2341         | 2025-11-19 15:48:57 | main.py

🧑 You: can you go to input_csvs folder

🤖 Agent: Changed working directory to /Users/fahimeh/Projects/csv_consolidation/input_csvs

🧑 You: what is there

🤖 Agent: Type | Size (Bytes) | Modified Date/Time      | Name
------------------------------------------------------------
FILE | 8486         | 2025-10-15 16:12:42 | sanandaj.csv
FILE | 6148         | 2025-10-15 16:15:27 | .DS_Store
FILE | 5831         | 2025-10-15 16:12:42 | tiran.csv
FILE | 5483         | 2025-10-15 16:12:42 | zahedan.csv
DIR  |              | 2025-10-15 16:13:08 | cleaned_csvs
FILE | 6543         | 2025-10-15 16:12:42 | shahroud.csv
FILE | 2637         | 2025-10-15 16:12:42 | shahrekord.csv
FILE | 5957         | 2025-10-15 16:12:42 | babol.csv
FILE | 4144         | 2025-10-15 16:12:42 | neyshabour_javad.csv
FILE | 12559        | 2025-10-15 16:12:42 | khoy.csv
FILE | 21989        | 2025-10-15 16:12:42 | jiroft.csv
FILE | 4133         | 2025-10-15 16:12:42 | arak.csv
FILE | 2094         | 2025-10-15 16:12:42 | naeen.csv
FILE | 24028        | 2025-10-15 16:12:42 | bonab.csv
FILE | 4444         | 2025-10-15 16:12:42 | boushehr.csv
DIR  |              | 2025-10-15 16:14:41 | consolidated_csv
FILE | 8601         | 2025-10-15 16:12:42 | ahzav_javad.csv
FILE | 3373         | 2025-10-15 16:12:42 | neyshabour_maryam.csv
FILE | 17621        | 2025-10-15 16:12:42 | shiraz.csv
FILE | 4080         | 2025-10-15 16:12:42 | Ahvaz_golshan.csv
FILE | 3688         | 2025-10-15 16:12:42 | bam.csv

🧑 You: can you list only the csv files in that folder?

🤖 Agent: ['Ahvaz_golshan.csv', 'ahzav_javad.csv', 'arak.csv', 'babol.csv', 'bam.csv', 'bonab.csv', 'boushehr.csv', 'jiroft.csv', 'khoy.csv', 'naeen.csv', 'neyshabour_javad.csv', 'neyshabour_maryam.csv', 'sanandaj.csv', 'shahrekord.csv', 'shahroud.csv', 'shiraz.csv', 'tiran.csv', 'zahedan.csv']

🧑 You: how many csvs from neyshabour we have

🤖 Agent: Found 1 CSV files related to center 'neyshabour_javad' in current directory.

🧑 You: how many from naeen

🤖 Agent: Found 1 CSV files related to center 'naeen' in current directory.

🧑 You: list all the files from naeen

🤖 Agent: ['naeen.csv']

🧑 You: read the files from naeen

🤖 Agent: نام مرکز → نام مرکز (exact match)
نوع دوره → نوع دوره (لطفا مشخص کنید: تقویتی - ورزشی - هنری - مهارتی و یا سایر) (100.0)
نام دقیق دوره → نام دقیق دوره (exact match)
تاریخ شروع دوره → تاریخ شروع دوره (exact match)
تاریخ پایان دوره → تاریخ پایان دوره (exact match)
نام مددجویان شرکت کننده → نام و نام خانوادگی مددجویان شرکت کننده (100.0)
نام معلم → نام و نام خانوادگی معلم (via synonym)
تعداد مددجویان شرکت کننده → تعداد مددجویان شرکت کننده (exact match)
تعداد جلسات دوره → تعداد جلسات دوره (exact match)
هزینه پرداختی به معلم → مبلغ واریزی به معلم (لطفا فقط عدد مبلغ را به تومان  و به انگلیسی وارد کنید) (via synonym)
هزینه های اضافی → هزینه های اضافی (لطفا فقط عدد مبلغ را به تومان  و به انگلیسی وارد کنید) (100.0)
توضیحات → توضیح هزینه های اضافی (via synonym)
🧑 You: clean the files from naeen

🤖 Agent: ✅ Cleaned file saved as: cleaned_csvs/naeen.csv

🔍 Preview:
نام مرکز نوع دوره      نام دقیق دوره تاریخ شروع دوره تاریخ پایان دوره               نام مددجویان شرکت کننده                  نام معلم  تعداد مددجویان شرکت کننده  تعداد جلسات دوره  هزینه پرداختی به معلم  هزینه های اضافی            توضیحات
   naeen    ورزشی آموزش شنا و فوتسال            None             None  مهدی زمانی مهر،سهیل قاسمی،سينا قاسمی علیرضا محبی فر،عباس صارمی                       12.0              30.0             19000000.0        5000000.0 ایاب وذهاب فرزندان
   naeen      nan                nan            None             None ابوالفضل کیان پور،عبدالمحسن دهقانیان،                       nan                        NaN               NaN                    NaN              NaN                NaN
   naeen      nan                nan            None             None           متین رمضانی،علی اصغر احسان،                       nan                        NaN               NaN                    NaN              NaN                NaN

🧑 You: thanks bye
🤖 Agent: Task complete. Terminating.
Terminating...

Agent has decided to terminate.


```


**Congratulations!** You now have a working AI agent that can:
- Understand natural language requests
- Choose appropriate tools
- Execute file operations safely
- Maintain conversation context
- Handle errors gracefully



## Summary:

1. GAME.py
	•	Core classes: Goal, Action, ActionRegistry, Memory, Environment.
	•	Defines the structure for goals, actions, storing memory, and executing actions safely.

2. agent.py
	•	Wraps everything into an Agent.
	•	Loops through:
	1.	Constructing a prompt from goals, memory, and actions.
	2.	Sending it to the LLM (generate_response).
	3.	Parsing the LLM’s response to choose an action.
	4.	Executing that action in the environment.
	5.	Updating memory.
	6.	Checking if the agent should terminate.

3. language.py
	•	Translates the agent’s goals, memory, and available actions into a prompt the LLM can understand.
	•	Handles parsing the LLM’s response into tool calls (parse_response).

4. response_generator.py
	•	Calls the LLM (Cohere via LangChain) with a prompt.
	•	Handles tool/function calls, formats the result, applies timeout handling.
	•	Returns either text or a JSON representing which tool to call and its arguments.

5. tool_registry.py
	•	Dynamically registers Python functions as tools for the agent.
	•	Provides metadata (parameters, JSON schema, description, terminal flag, tags).
	•	Decorator @register_tool makes a function available to the agent.

⸻

Full flow:
	1.	User input: "List all CSV files in input_csvs".
	2.	Agent: set_current_task → memory updated.
	3.	Prompt built: AgentLanguage.construct_prompt() gathers goals, memory, actions.
	4.	LLM call: generate_response() → returns JSON like:
     5.	Agent parses response: get_action() → gets the corresponding Action.
	6.	Action executed: Environment.execute_action() → runs the Python function safely.
	7.	Memory updated: stores both the agent’s reasoning and environment result.
	8.	Check termination: repeat until either the agent calls terminate or max iterations reached.
