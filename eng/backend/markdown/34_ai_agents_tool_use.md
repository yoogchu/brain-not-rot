# Chapter 34: AI Agents & Tool Use

## Why AI Agents?

Without agentic systems:

```
User: "Find all customers who churned last month and send them offers"

Without agents:
- Engineer writes custom script
- Queries database manually
- Crafts email template
- Runs send script
- Monitors results manually
Time: 2-3 hours

With agents:
- Agent understands intent
- Queries database via SQL tool
- Generates personalized emails via LLM
- Sends via email tool
- Reports results
Time: 30 seconds
```

But naive agent implementations fail in production:

```
Normal request: "Summarize Q3 sales"
Agent behavior:
1. Calls get_sales_data() → $0.02
2. Returns summary → SUCCESS

Edge case: "Compare Q1-Q4 sales for all regions and products"
Naive agent behavior:
1. Calls get_sales_data(Q1, region1, product1) → 1000x in loop
2. Token limit exceeded after 200 calls
3. Cost: $2000, Time: 10 minutes, Result: FAILURE

Production-ready agents need:
- Guardrails (cost, latency, tool call limits)
- Error recovery and retries
- Observable execution traces
- Deterministic testing
```

---

## The ReAct Pattern

**The Problem:**
LLMs are trained to generate text, not execute actions. They hallucinate tool results instead of calling real tools.

**How It Works:**

ReAct (Reasoning + Acting) interleaves thinking and tool use:

```
┌─────────────────────────────────────────────────┐
│              User Query                          │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Agent Loop (until done or max iterations)      │
│                                                  │
│  ┌──────────────┐        ┌──────────────┐       │
│  │  Thought:    │        │  Action:     │       │
│  │  What do I   │───────►│  Call tool   │       │
│  │  need next?  │        │  with args   │       │
│  └──────────────┘        └──────────────┘       │
│                                 │                │
│                                 ▼                │
│                          ┌──────────────┐        │
│                          │ Observation: │        │
│  ┌──────────────┐        │ Tool result  │        │
│  │ Final Answer │◄───────┤              │        │
│  └──────────────┘        └──────────────┘        │
│         │                       │                │
│         │                       ▼                │
│         │              Loop back (next thought)  │
│         │                                        │
└─────────┼────────────────────────────────────────┘
          │
          ▼
    Return to user
```

**Implementation:**

```python
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional
import json

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, str]  # {param_name: description}

class ReActAgent:
    def __init__(self, llm_client, tools: List[Tool], max_iterations=10):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}\n  Parameters: {tool.parameters}"
            for tool in self.tools.values()
        ])

        return f"""You are an AI agent that can use tools to answer questions.

Available tools:
{tool_descriptions}

Use this format:

Thought: [your reasoning about what to do next]
Action: [tool name]
Action Input: {{"param1": "value1", "param2": "value2"}}
Observation: [tool result will be inserted here]

When you have the final answer:
Thought: I now know the final answer
Final Answer: [your answer to the user]
"""

    def run(self, user_query: str) -> Dict[str, Any]:
        """Execute agent loop using ReAct pattern"""
        conversation = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_query}
        ]

        trace = []  # Execution trace for debugging

        for iteration in range(self.max_iterations):
            # Get next action from LLM
            response = self.llm.chat(conversation)
            agent_output = response.content

            trace.append({
                "iteration": iteration,
                "agent_output": agent_output
            })

            # Check if we have final answer
            if "Final Answer:" in agent_output:
                final_answer = agent_output.split("Final Answer:")[1].strip()
                return {
                    "answer": final_answer,
                    "trace": trace,
                    "iterations": iteration + 1
                }

            # Parse action and execute tool
            try:
                action, action_input = self._parse_action(agent_output)
                observation = self._execute_tool(action, action_input)

                trace[-1]["action"] = action
                trace[-1]["action_input"] = action_input
                trace[-1]["observation"] = observation

                # Add observation to conversation
                conversation.append({
                    "role": "assistant",
                    "content": agent_output
                })
                conversation.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

            except Exception as e:
                # Tool execution failed
                error_msg = f"Error: {str(e)}"
                trace[-1]["error"] = error_msg
                conversation.append({
                    "role": "user",
                    "content": error_msg
                })

        # Max iterations reached
        return {
            "answer": None,
            "error": "Max iterations reached",
            "trace": trace
        }

    def _parse_action(self, agent_output: str) -> tuple[str, Dict]:
        """Extract action and inputs from agent output"""
        # Simple parsing - production code needs robust parsing
        lines = agent_output.strip().split('\n')

        action = None
        action_input = None

        for line in lines:
            if line.startswith("Action:"):
                action = line.split("Action:")[1].strip()
            elif line.startswith("Action Input:"):
                input_str = line.split("Action Input:")[1].strip()
                action_input = json.loads(input_str)

        if not action or not action_input:
            raise ValueError("Could not parse action from agent output")

        return action, action_input

    def _execute_tool(self, tool_name: str, inputs: Dict) -> str:
        """Execute tool and return observation"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self.tools[tool_name]
        result = tool.function(**inputs)
        return str(result)


# Example usage
def search_database(query: str) -> List[Dict]:
    """Search customer database"""
    # Mock implementation
    return [
        {"customer_id": "123", "name": "Alice", "status": "churned"},
        {"customer_id": "456", "name": "Bob", "status": "churned"}
    ]

def send_email(customer_id: str, template: str) -> bool:
    """Send email to customer"""
    # Mock implementation
    print(f"Sending email to {customer_id}: {template}")
    return True

# Define tools
tools = [
    Tool(
        name="search_database",
        description="Search customer database by query",
        function=search_database,
        parameters={"query": "SQL-like query string"}
    ),
    Tool(
        name="send_email",
        description="Send email to customer",
        function=send_email,
        parameters={
            "customer_id": "Customer ID",
            "template": "Email content"
        }
    )
]

agent = ReActAgent(llm_client, tools, max_iterations=10)
result = agent.run("Find churned customers and send them a 20% discount offer")
print(result["answer"])
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Transparency | Full reasoning trace visible | Verbose, high token usage |
| Reliability | Can retry on errors | No guarantees of completion |
| Cost | Only pay per action | Many LLM calls in loop |
| Latency | Sequential execution | Slow for multi-step tasks |

**When to use:** Complex tasks requiring multi-step reasoning, need for interpretability

**When NOT to use:** Simple single-tool calls, latency-critical paths, high-volume requests

---

## Function Calling APIs

**The Problem:**
Parsing tool calls from LLM text is brittle. LLMs might return invalid JSON, misspell function names, or hallucinate parameters.

**How It Works:**

Modern LLMs support native function calling with structured outputs:

```
┌──────────────────────────────────────────────┐
│           User: "Get weather in SF"          │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  LLM with function definitions               │
│                                              │
│  Functions:                                  │
│  - get_weather(location: str, units: str)    │
│  - search_web(query: str)                    │
│  - send_email(to: str, body: str)            │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Structured function call:                   │
│  {                                           │
│    "name": "get_weather",                    │
│    "arguments": {                            │
│      "location": "San Francisco",            │
│      "units": "celsius"                      │
│    }                                         │
│  }                                           │
└──────────────────────────────────────────────┘
                     │
                     ▼
         Execute function → Return result
```

**Implementation with OpenAI API:**

```python
from openai import OpenAI
from typing import List, Dict, Any
import json

class FunctionCallingAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.tools_registry = {}

    def register_tool(self, func: Callable, schema: Dict):
        """Register a tool with its OpenAI function schema"""
        self.tools_registry[schema["name"]] = func

    def run(self, messages: List[Dict], tools: List[Dict]) -> str:
        """Run agent with function calling"""
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            tools=tools,
            tool_choice="auto"  # Let model decide when to call
        )

        message = response.choices[0].message

        # If no tool calls, return text response
        if not message.tool_calls:
            return message.content

        # Execute all tool calls
        messages.append(message)

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Execute registered function
            if function_name in self.tools_registry:
                result = self.tools_registry[function_name](**function_args)

                # Add result to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(result)
                })

        # Get final response with tool results
        final_response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )

        return final_response.choices[0].message.content


# Define tool schema (OpenAI format)
get_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }
    }
}

def get_weather(location: str, units: str = "celsius") -> Dict:
    """Actual function implementation"""
    # Mock implementation
    return {
        "location": location,
        "temperature": 18,
        "units": units,
        "condition": "partly cloudy"
    }

# Usage
agent = FunctionCallingAgent(api_key="sk-...")
agent.register_tool(get_weather, get_weather_schema["function"])

messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
result = agent.run(messages, tools=[get_weather_schema])
print(result)  # "The current weather in Tokyo is partly cloudy, 18°C"
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Reliability | Structured outputs, validated | Vendor lock-in |
| Error handling | Type checking built-in | Still need runtime validation |
| Development speed | Less parsing code | API-specific schemas |
| Token efficiency | No verbose prompts | Function schemas use tokens |

**When to use:** Production systems, need reliability, using OpenAI/Anthropic/similar

**When NOT to use:** Open source models without function calling, need full control

---

## Agent Memory Systems

**The Problem:**
Agents need to remember past interactions to maintain context, but LLM context windows are limited and expensive.

**How It Works:**

Three types of memory:

```
┌─────────────────────────────────────────────────┐
│                 Memory Hierarchy                 │
│                                                  │
│  ┌────────────────────────────────────┐         │
│  │  Working Memory (Current Context)  │         │
│  │  - Last N messages                 │         │
│  │  - Current tool results            │         │
│  │  Fast, expensive (in context)      │         │
│  └────────────────────────────────────┘         │
│                    │                             │
│                    ▼                             │
│  ┌────────────────────────────────────┐         │
│  │  Short-term Memory (Session)       │         │
│  │  - Conversation history            │         │
│  │  - Recent decisions                │         │
│  │  Redis/in-memory cache             │         │
│  └────────────────────────────────────┘         │
│                    │                             │
│                    ▼                             │
│  ┌────────────────────────────────────┐         │
│  │  Long-term Memory (Facts)          │         │
│  │  - User preferences                │         │
│  │  - Domain knowledge                │         │
│  │  Vector database                   │         │
│  └────────────────────────────────────┘         │
└─────────────────────────────────────────────────┘
```

**Implementation:**

```python
from typing import List, Dict, Optional
import redis
from datetime import datetime, timedelta

class AgentMemory:
    def __init__(
        self,
        redis_client: redis.Redis,
        vector_db,
        max_working_memory: int = 10
    ):
        self.redis = redis_client
        self.vector_db = vector_db
        self.max_working_memory = max_working_memory

    def get_context(
        self,
        session_id: str,
        user_query: str
    ) -> Dict[str, Any]:
        """Build context from all memory layers"""

        # 1. Get working memory (recent messages)
        working_memory = self._get_working_memory(session_id)

        # 2. Get relevant long-term memories via vector search
        long_term_memory = self._search_long_term_memory(
            session_id,
            user_query,
            top_k=3
        )

        # 3. Get session metadata (user preferences, state)
        session_metadata = self._get_session_metadata(session_id)

        return {
            "working_memory": working_memory,
            "long_term_memory": long_term_memory,
            "session_metadata": session_metadata
        }

    def _get_working_memory(self, session_id: str) -> List[Dict]:
        """Get recent conversation from Redis"""
        key = f"session:{session_id}:messages"
        messages = self.redis.lrange(key, -self.max_working_memory, -1)
        return [json.loads(msg) for msg in messages]

    def _search_long_term_memory(
        self,
        session_id: str,
        query: str,
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant facts from vector DB"""
        results = self.vector_db.search(
            query=query,
            filter={"session_id": session_id},
            top_k=top_k
        )
        return results

    def _get_session_metadata(self, session_id: str) -> Dict:
        """Get session state and preferences"""
        key = f"session:{session_id}:metadata"
        metadata = self.redis.get(key)
        return json.loads(metadata) if metadata else {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        save_to_longterm: bool = False
    ):
        """Add message to working memory"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add to Redis (short-term)
        key = f"session:{session_id}:messages"
        self.redis.rpush(key, json.dumps(message))
        self.redis.expire(key, timedelta(hours=24))

        # Optionally save important facts to long-term memory
        if save_to_longterm:
            self.vector_db.add(
                text=content,
                metadata={
                    "session_id": session_id,
                    "role": role,
                    "timestamp": message["timestamp"]
                }
            )

    def update_session_metadata(
        self,
        session_id: str,
        updates: Dict
    ):
        """Update session metadata (preferences, state)"""
        key = f"session:{session_id}:metadata"

        # Get existing
        existing = self.redis.get(key)
        metadata = json.loads(existing) if existing else {}

        # Merge updates
        metadata.update(updates)

        # Save back
        self.redis.set(
            key,
            json.dumps(metadata),
            ex=timedelta(days=30)
        )


# Usage in agent
class StatefulAgent:
    def __init__(self, llm, memory: AgentMemory):
        self.llm = llm
        self.memory = memory

    def run(self, session_id: str, user_query: str) -> str:
        # Get context from all memory layers
        context = self.memory.get_context(session_id, user_query)

        # Build prompt with context
        messages = [
            {"role": "system", "content": self._build_system_prompt(context)},
            *context["working_memory"],
            {"role": "user", "content": user_query}
        ]

        # Get response
        response = self.llm.chat(messages)

        # Save to memory
        self.memory.add_message(session_id, "user", user_query)
        self.memory.add_message(session_id, "assistant", response.content)

        return response.content

    def _build_system_prompt(self, context: Dict) -> str:
        """Include relevant long-term memories in system prompt"""
        long_term_facts = "\n".join([
            f"- {mem['text']}"
            for mem in context["long_term_memory"]
        ])

        metadata = context["session_metadata"]

        return f"""You are a helpful assistant.

User preferences:
{json.dumps(metadata.get('preferences', {}), indent=2)}

Relevant past information:
{long_term_facts}
"""
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Context window usage | Only relevant info in context | Retrieval adds latency |
| Cost | Cheaper than full history | Vector DB costs |
| Accuracy | Access to all past data | Retrieval can miss context |
| Complexity | Modular architecture | Multiple systems to maintain |

**When to use:** Long conversations, multi-session agents, personalization

**When NOT to use:** Stateless APIs, single-turn interactions, privacy-sensitive data

---

## Multi-Agent Architectures

**The Problem:**
Single agents struggle with complex tasks requiring different skills (research, writing, coding, validation).

**How It Works:**

Specialized agents collaborate on tasks:

```
┌─────────────────────────────────────────────────┐
│         Orchestrator / Supervisor                │
│     (Routes tasks to specialist agents)          │
└─────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Researcher  │  │    Writer    │  │   Reviewer   │
│              │  │              │  │              │
│ Tools:       │  │ Tools:       │  │ Tools:       │
│ - Web search │  │ - Templates  │  │ - Validators │
│ - Database   │  │ - Grammar    │  │ - Fact check │
└──────────────┘  └──────────────┘  └──────────────┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
                 ┌──────────────┐
                 │ Shared State │
                 │  (Memory)    │
                 └──────────────┘
```

**Implementation:**

```python
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"

@dataclass
class AgentMessage:
    from_agent: AgentRole
    to_agent: AgentRole
    task: str
    context: Dict
    result: Optional[str] = None

class SpecializedAgent:
    def __init__(self, role: AgentRole, llm, tools: List[Tool]):
        self.role = role
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}

    def execute(self, task: str, context: Dict) -> str:
        """Execute task with available tools"""
        system_prompt = self._get_role_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_task(task, context)}
        ]

        # Simple function calling loop
        for _ in range(5):  # Max 5 tool calls
            response = self.llm.chat(
                messages,
                tools=self._get_tool_schemas()
            )

            if not response.tool_calls:
                return response.content

            # Execute tools and continue
            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id
                })

        return "Task incomplete - max iterations reached"

    def _get_role_prompt(self) -> str:
        """Role-specific system prompts"""
        prompts = {
            AgentRole.RESEARCHER: """You are a research specialist.
Your job is to gather information from available sources.
Be thorough and cite sources.""",

            AgentRole.WRITER: """You are a writing specialist.
Create clear, engaging content based on research.
Follow style guidelines provided.""",

            AgentRole.REVIEWER: """You are a quality reviewer.
Check facts, grammar, and completeness.
Provide specific feedback."""
        }
        return prompts.get(self.role, "")

    def _format_task(self, task: str, context: Dict) -> str:
        """Format task with context"""
        context_str = "\n".join([
            f"{k}: {v}" for k, v in context.items()
        ])
        return f"{task}\n\nContext:\n{context_str}"

    def _execute_tool(self, tool_call) -> str:
        """Execute a tool and return result"""
        tool = self.tools.get(tool_call.function.name)
        if not tool:
            return f"Error: Unknown tool {tool_call.function.name}"

        args = json.loads(tool_call.function.arguments)
        result = tool.function(**args)
        return json.dumps(result)

    def _get_tool_schemas(self) -> List[Dict]:
        """Get OpenAI tool schemas"""
        return [self._tool_to_schema(tool) for tool in self.tools.values()]

    def _tool_to_schema(self, tool: Tool) -> Dict:
        """Convert Tool to OpenAI schema"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param: {"type": "string", "description": desc}
                        for param, desc in tool.parameters.items()
                    },
                    "required": list(tool.parameters.keys())
                }
            }
        }


class MultiAgentOrchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.agents: Dict[AgentRole, SpecializedAgent] = {}
        self.shared_state = {}

    def register_agent(self, agent: SpecializedAgent):
        """Register a specialized agent"""
        self.agents[agent.role] = agent

    def execute_workflow(
        self,
        task: str,
        workflow: List[AgentRole]
    ) -> Dict[str, Any]:
        """Execute multi-agent workflow"""
        context = {"original_task": task}
        results = {}

        for agent_role in workflow:
            if agent_role not in self.agents:
                raise ValueError(f"Agent not registered: {agent_role}")

            agent = self.agents[agent_role]

            # Execute this stage
            result = agent.execute(
                task=self._create_stage_task(agent_role, task, context),
                context=context
            )

            # Save result and update context
            results[agent_role.value] = result
            context[f"{agent_role.value}_output"] = result

        return {
            "final_result": result,
            "all_results": results,
            "context": context
        }

    def _create_stage_task(
        self,
        role: AgentRole,
        original_task: str,
        context: Dict
    ) -> str:
        """Create stage-specific task description"""
        if role == AgentRole.RESEARCHER:
            return f"Research the following: {original_task}"
        elif role == AgentRole.WRITER:
            return f"Write content based on research about: {original_task}"
        elif role == AgentRole.REVIEWER:
            return "Review the written content for accuracy and quality"
        return original_task


# Usage
orchestrator = MultiAgentOrchestrator(llm_client)

# Register specialized agents
researcher = SpecializedAgent(
    role=AgentRole.RESEARCHER,
    llm=llm_client,
    tools=[web_search_tool, database_tool]
)
writer = SpecializedAgent(
    role=AgentRole.WRITER,
    llm=llm_client,
    tools=[template_tool, grammar_tool]
)
reviewer = SpecializedAgent(
    role=AgentRole.REVIEWER,
    llm=llm_client,
    tools=[fact_check_tool, validator_tool]
)

orchestrator.register_agent(researcher)
orchestrator.register_agent(writer)
orchestrator.register_agent(reviewer)

# Execute workflow
result = orchestrator.execute_workflow(
    task="Write a blog post about AI safety",
    workflow=[
        AgentRole.RESEARCHER,
        AgentRole.WRITER,
        AgentRole.REVIEWER
    ]
)

print(result["final_result"])
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Quality | Specialized expertise | More complex coordination |
| Debuggability | Clear stage boundaries | Harder to trace errors |
| Cost | Only use needed agents | Multiple LLM calls |
| Latency | Can parallelize | Sequential by default |

**When to use:** Complex workflows, need quality checks, different skills required

**When NOT to use:** Simple tasks, tight latency budgets, cost-sensitive applications

---

## Guardrails and Safety

**The Problem:**
Agents can go off the rails - infinite loops, excessive costs, dangerous actions, hallucinated tools.

**How It Works:**

Multiple layers of protection:

```python
from typing import Callable, Dict, Any
import time
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class GuardrailViolation(Exception):
    """Raised when guardrail is violated"""
    rule: str
    details: str

class AgentGuardrails:
    def __init__(
        self,
        max_iterations: int = 10,
        max_cost_cents: int = 100,  # $1.00
        max_latency_seconds: int = 30,
        max_tool_calls: int = 20,
        allowed_tools: Optional[List[str]] = None
    ):
        self.max_iterations = max_iterations
        self.max_cost_cents = max_cost_cents
        self.max_latency_seconds = max_latency_seconds
        self.max_tool_calls = max_tool_calls
        self.allowed_tools = set(allowed_tools) if allowed_tools else None

        # Runtime tracking
        self.iteration_count = 0
        self.total_cost_cents = 0
        self.start_time = None
        self.tool_call_count = 0

    def start_run(self):
        """Initialize run tracking"""
        self.iteration_count = 0
        self.total_cost_cents = 0
        self.start_time = time.time()
        self.tool_call_count = 0

    def check_iteration(self):
        """Check iteration limit"""
        self.iteration_count += 1
        if self.iteration_count > self.max_iterations:
            raise GuardrailViolation(
                rule="max_iterations",
                details=f"Exceeded {self.max_iterations} iterations"
            )

    def check_cost(self, cost_cents: int):
        """Check cost limit"""
        self.total_cost_cents += cost_cents
        if self.total_cost_cents > self.max_cost_cents:
            raise GuardrailViolation(
                rule="max_cost",
                details=f"Exceeded ${self.max_cost_cents/100:.2f} budget"
            )

    def check_latency(self):
        """Check latency limit"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_latency_seconds:
                raise GuardrailViolation(
                    rule="max_latency",
                    details=f"Exceeded {self.max_latency_seconds}s timeout"
                )

    def check_tool_call(self, tool_name: str):
        """Check tool call limits and allowlist"""
        # Check total tool calls
        self.tool_call_count += 1
        if self.tool_call_count > self.max_tool_calls:
            raise GuardrailViolation(
                rule="max_tool_calls",
                details=f"Exceeded {self.max_tool_calls} tool calls"
            )

        # Check tool allowlist
        if self.allowed_tools and tool_name not in self.allowed_tools:
            raise GuardrailViolation(
                rule="tool_allowlist",
                details=f"Tool '{tool_name}' not in allowlist"
            )

    def validate_tool_args(
        self,
        tool_name: str,
        args: Dict,
        validators: Dict[str, Callable]
    ):
        """Validate tool arguments against custom rules"""
        if tool_name in validators:
            validator = validators[tool_name]
            if not validator(args):
                raise GuardrailViolation(
                    rule="tool_args_validation",
                    details=f"Invalid arguments for {tool_name}: {args}"
                )


class SafeAgent:
    def __init__(
        self,
        llm,
        tools: List[Tool],
        guardrails: AgentGuardrails
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.guardrails = guardrails

        # Tool argument validators
        self.tool_validators = {}

    def register_validator(
        self,
        tool_name: str,
        validator: Callable[[Dict], bool]
    ):
        """Register custom validator for tool arguments"""
        self.tool_validators[tool_name] = validator

    def run(self, user_query: str) -> Dict[str, Any]:
        """Run agent with guardrails"""
        self.guardrails.start_run()

        messages = [{"role": "user", "content": user_query}]

        try:
            while True:
                # Check limits before iteration
                self.guardrails.check_iteration()
                self.guardrails.check_latency()

                # Get LLM response
                response = self.llm.chat(
                    messages,
                    tools=self._get_tool_schemas()
                )

                # Track cost (example: $0.01 per 1000 tokens)
                tokens = response.usage.total_tokens
                cost_cents = int((tokens / 1000) * 1)
                self.guardrails.check_cost(cost_cents)

                # Check if done
                if not response.tool_calls:
                    return {
                        "answer": response.content,
                        "cost_cents": self.guardrails.total_cost_cents,
                        "iterations": self.guardrails.iteration_count,
                        "tool_calls": self.guardrails.tool_call_count
                    }

                # Execute tool calls with guardrails
                messages.append(response.message)

                for tool_call in response.tool_calls:
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    # Check guardrails
                    self.guardrails.check_tool_call(tool_name)
                    self.guardrails.check_latency()

                    # Validate arguments
                    self.guardrails.validate_tool_args(
                        tool_name,
                        args,
                        self.tool_validators
                    )

                    # Execute tool
                    result = self._safe_execute_tool(tool_name, args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

        except GuardrailViolation as e:
            return {
                "error": "guardrail_violation",
                "rule": e.rule,
                "details": e.details,
                "cost_cents": self.guardrails.total_cost_cents,
                "iterations": self.guardrails.iteration_count
            }
        except Exception as e:
            return {
                "error": "execution_error",
                "details": str(e),
                "cost_cents": self.guardrails.total_cost_cents
            }

    def _safe_execute_tool(self, tool_name: str, args: Dict) -> str:
        """Execute tool with error handling"""
        if tool_name not in self.tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            tool = self.tools[tool_name]
            result = tool.function(**args)
            return json.dumps({"result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})


# Example: SQL injection prevention
def validate_sql_query(args: Dict) -> bool:
    """Prevent SQL injection in database queries"""
    query = args.get("query", "")

    # Block dangerous keywords
    dangerous = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE"]
    query_upper = query.upper()

    for keyword in dangerous:
        if keyword in query_upper:
            return False

    return True

# Example: Cost control for expensive operations
def validate_bulk_email(args: Dict) -> bool:
    """Limit number of emails in single call"""
    recipient_count = len(args.get("recipients", []))
    return recipient_count <= 100  # Max 100 emails per call

# Usage
guardrails = AgentGuardrails(
    max_iterations=10,
    max_cost_cents=50,  # $0.50 max
    max_latency_seconds=30,
    max_tool_calls=15,
    allowed_tools=["search_database", "send_email"]
)

agent = SafeAgent(llm_client, tools, guardrails)
agent.register_validator("search_database", validate_sql_query)
agent.register_validator("send_email", validate_bulk_email)

result = agent.run("Send emails to all users")
print(result)
```

**Guardrail Categories:**

| Category | Examples | Implementation |
|----------|----------|----------------|
| Resource limits | Cost, latency, iterations | Counter tracking |
| Tool restrictions | Allowlists, dangerous ops | Pre-execution checks |
| Input validation | SQL injection, prompt injection | Regex, validators |
| Output filtering | PII, toxic content | Post-processing |

**When to use:** All production agents, user-facing applications, high-stakes operations

**When NOT to use:** Controlled environments, internal testing, fully trusted users

---

## Evaluation and Testing

**The Problem:**
Agents are non-deterministic. Traditional unit tests fail. How do you know if changes improve or break behavior?

**How It Works:**

Multi-faceted evaluation approach:

```python
from typing import List, Dict, Callable
import json
from dataclasses import dataclass

@dataclass
class TestCase:
    id: str
    input: str
    expected_output: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    expected_outcome: Optional[str] = None  # "success" | "error"
    max_cost_cents: Optional[int] = None
    max_latency_seconds: Optional[int] = None

@dataclass
class EvalResult:
    test_id: str
    passed: bool
    actual_output: str
    actual_tools: List[str]
    cost_cents: int
    latency_seconds: float
    errors: List[str]

class AgentEvaluator:
    def __init__(self, agent, llm_judge=None):
        self.agent = agent
        self.llm_judge = llm_judge  # LLM for semantic evaluation

    def evaluate(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run evaluation suite"""
        results = []

        for test in test_cases:
            result = self._run_test(test)
            results.append(result)

        return self._aggregate_results(results)

    def _run_test(self, test: TestCase) -> EvalResult:
        """Run single test case"""
        start_time = time.time()
        errors = []

        # Execute agent
        try:
            output = self.agent.run(test.input)
            actual_output = output.get("answer", "")
            actual_tools = output.get("tools_used", [])
            cost_cents = output.get("cost_cents", 0)
        except Exception as e:
            actual_output = ""
            actual_tools = []
            cost_cents = 0
            errors.append(f"Execution error: {str(e)}")

        latency = time.time() - start_time

        # Check expectations
        passed = True

        # 1. Check expected outcome
        if test.expected_outcome:
            if test.expected_outcome == "success" and errors:
                passed = False
                errors.append("Expected success but got errors")
            elif test.expected_outcome == "error" and not errors:
                passed = False
                errors.append("Expected error but succeeded")

        # 2. Check expected tools
        if test.expected_tools:
            if set(actual_tools) != set(test.expected_tools):
                passed = False
                errors.append(
                    f"Tool mismatch: expected {test.expected_tools}, "
                    f"got {actual_tools}"
                )

        # 3. Check cost budget
        if test.max_cost_cents and cost_cents > test.max_cost_cents:
            passed = False
            errors.append(
                f"Cost exceeded: {cost_cents}¢ > {test.max_cost_cents}¢"
            )

        # 4. Check latency budget
        if test.max_latency_seconds and latency > test.max_latency_seconds:
            passed = False
            errors.append(
                f"Latency exceeded: {latency:.2f}s > "
                f"{test.max_latency_seconds}s"
            )

        # 5. Check semantic correctness (if expected output provided)
        if test.expected_output:
            semantic_match = self._check_semantic_match(
                test.expected_output,
                actual_output
            )
            if not semantic_match:
                passed = False
                errors.append("Semantic mismatch")

        return EvalResult(
            test_id=test.id,
            passed=passed,
            actual_output=actual_output,
            actual_tools=actual_tools,
            cost_cents=cost_cents,
            latency_seconds=latency,
            errors=errors
        )

    def _check_semantic_match(
        self,
        expected: str,
        actual: str
    ) -> bool:
        """Use LLM to judge semantic equivalence"""
        if not self.llm_judge:
            # Fallback to exact match
            return expected.strip() == actual.strip()

        prompt = f"""Compare these two answers:

Expected: {expected}

Actual: {actual}

Are they semantically equivalent? Answer ONLY 'yes' or 'no'.
"""

        response = self.llm_judge.chat([
            {"role": "user", "content": prompt}
        ])

        return "yes" in response.content.lower()

    def _aggregate_results(self, results: List[EvalResult]) -> Dict:
        """Aggregate evaluation results"""
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        total_cost = sum(r.cost_cents for r in results)
        avg_latency = sum(r.latency_seconds for r in results) / total

        return {
            "pass_rate": passed / total if total > 0 else 0,
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "total_cost_cents": total_cost,
            "avg_latency_seconds": avg_latency,
            "failed_tests": [
                {
                    "id": r.test_id,
                    "errors": r.errors
                }
                for r in results if not r.passed
            ]
        }


# Example test suite
test_suite = [
    TestCase(
        id="basic_query",
        input="What is 2+2?",
        expected_output="4",
        expected_tools=[],  # No tools needed
        max_cost_cents=5,
        max_latency_seconds=2
    ),
    TestCase(
        id="database_lookup",
        input="Find all customers in California",
        expected_tools=["search_database"],
        expected_outcome="success",
        max_cost_cents=10
    ),
    TestCase(
        id="invalid_sql_blocked",
        input="DROP TABLE users",
        expected_outcome="error",  # Should be blocked by guardrails
    ),
    TestCase(
        id="multi_step_task",
        input="Find churned customers and email them",
        expected_tools=["search_database", "send_email"],
        expected_outcome="success",
        max_cost_cents=50,
        max_latency_seconds=10
    )
]

# Run evaluation
evaluator = AgentEvaluator(agent, llm_judge=judge_llm)
results = evaluator.evaluate(test_suite)

print(f"Pass rate: {results['pass_rate']:.1%}")
print(f"Total cost: ${results['total_cost_cents']/100:.2f}")
print(f"Avg latency: {results['avg_latency_seconds']:.2f}s")

if results['failed_tests']:
    print("\nFailed tests:")
    for fail in results['failed_tests']:
        print(f"  {fail['id']}: {fail['errors']}")
```

**Evaluation Dimensions:**

| Dimension | What to measure | How |
|-----------|----------------|-----|
| Correctness | Right answer | LLM judge, exact match |
| Tool usage | Right tools called | Tool trace comparison |
| Efficiency | Token/cost usage | Budget tracking |
| Latency | Response time | Timing |
| Reliability | Error rate | Success/failure ratio |
| Safety | Guardrail violations | Violation logging |

**When to use:** All production agents, before deployments, A/B testing variants

**When NOT to use:** Exploratory prototyping (but add before production)

---

## Framework Comparison

| Framework | Strengths | Weaknesses | Best For |
|-----------|-----------|------------|----------|
| **LangChain** | Rich ecosystem, many integrations, active community | Heavy abstraction, complex API, version instability | Rapid prototyping, standard use cases |
| **LlamaIndex** | Excellent RAG support, data connectors, query engines | Less flexible for custom agents, opinionated | Document Q&A, knowledge bases |
| **AutoGPT** | Autonomous task execution, ambitious goals | Unreliable, expensive loops, hard to control | Experimental, research |
| **Custom** | Full control, minimal dependencies, debuggable | More code to write, maintain | Production systems, specific requirements |

**Framework decision tree:**

```
Need RAG + document processing?
  └─► LlamaIndex

Standard chatbot + tools?
  └─► LangChain

Production system with specific requirements?
  └─► Custom implementation

Research / experimentation?
  └─► AutoGPT or custom
```

---

## Key Concepts Checklist

- [ ] Explain ReAct pattern (reasoning + acting loop)
- [ ] Implement function calling with structured outputs
- [ ] Design memory system (working, short-term, long-term)
- [ ] Architect multi-agent workflows with specialization
- [ ] Implement guardrails (cost, latency, tool restrictions)
- [ ] Build evaluation framework with LLM judges
- [ ] Compare LangChain vs LlamaIndex vs custom agents
- [ ] Know when NOT to use agents (simple tasks, latency-critical)

---

## Practical Insights

**Cost control is critical:**
- Track costs per request in production
- Set hard limits: $0.10-$1.00 per agent run for most use cases
- Monitor token usage: context size grows quickly with memory
- Example: 10k requests/day * $0.50/request = $5,000/day = $150k/month
- Use cheaper models for tool selection, expensive for final generation

**Latency optimization:**
```python
# Bad: Sequential tool calls
for customer in customers:
    send_email(customer)  # 100 customers = 100 LLM calls

# Good: Batch operations
send_bulk_email(customers)  # 1 LLM call + 1 bulk tool call

# Best: Parallel execution
async def run_parallel():
    tasks = [agent.run(query) for query in queries]
    return await asyncio.gather(*tasks)
```

**Debugging agent behavior:**
- Always log full execution trace (thoughts, actions, observations)
- Use deterministic LLM settings (temperature=0) for debugging
- Create reproduction test cases for failures
- Monitor tool call patterns - loops indicate issues

```python
# Add detailed logging
logger.info("agent_iteration", {
    "iteration": i,
    "thought": thought,
    "action": action,
    "args": args,
    "observation": observation,
    "cost_so_far": cost,
    "latency": elapsed
})
```

**When NOT to use agents:**
- Simple single-tool calls - use direct function calling
- Latency < 1 second required - pre-compute or use rules
- Fully deterministic behavior needed - use traditional code
- Cost-sensitive high-volume - agents are expensive
- Regulated domains - explainability and audit trail challenging

**Error handling:**
- Tool failures are common - implement retries with exponential backoff
- LLMs hallucinate tool names - validate before execution
- Infinite loops happen - always set max iteration limits
- Graceful degradation: return partial results on timeout

```python
# Robust tool execution
def execute_tool_with_retry(tool_name, args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return tools[tool_name](**args)
        except ToolExecutionError as e:
            if attempt == max_retries - 1:
                return {"error": str(e), "recoverable": False}
            time.sleep(2 ** attempt)  # Exponential backoff
    return {"error": "Max retries exceeded"}
```

**Production deployment checklist:**
- [ ] Guardrails configured (cost, latency, iterations)
- [ ] All tools have input validation
- [ ] Dangerous operations require human-in-the-loop
- [ ] Execution traces logged for debugging
- [ ] Evaluation suite covers edge cases
- [ ] Monitoring alerts on cost spikes, error rates
- [ ] Fallback behavior for agent failures
