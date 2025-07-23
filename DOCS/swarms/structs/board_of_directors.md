# Board of Directors Swarm

The Board of Directors Swarm is an advanced multi-agent orchestration system that provides collective decision-making capabilities as an alternative to the single Director approach. This feature enables democratic and collaborative task management through a hierarchical board structure.

## Overview

The Board of Directors operates as a collective decision-making body that can be enabled manually through configuration. It provides a more democratic approach to task orchestration compared to the traditional single Director model.

### Key Features

- **Collective Decision Making**: Multiple board members collaborate to make decisions
- **Role-Based Governance**: Different board members have specific roles and responsibilities
- **Voting and Consensus**: Support for various decision-making mechanisms
- **Parallel Task Execution**: Efficient distribution and execution of tasks
- **Comprehensive Logging**: Detailed logging and monitoring capabilities
- **Async Support**: Full asynchronous operation support
- **Configuration Management**: Flexible configuration through environment variables and files

## Architecture

The Board of Directors Swarm follows a hierarchical architecture:

```
User Task
    ↓
Board of Directors Meeting
    ↓
Decision Making & Planning
    ↓
Task Distribution to Agents
    ↓
Parallel Task Execution
    ↓
Board Feedback & Iteration
```

## Quick Start

### Basic Usage

```python
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms.structs.agent import Agent

# Create worker agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Research analyst for data gathering",
    model_name="gpt-4o-mini"
)

writer = Agent(
    agent_name="Writer", 
    agent_description="Content writer for reports",
    model_name="gpt-4o-mini"
)

# Create Board of Directors swarm
board_swarm = BoardOfDirectorsSwarm(
    name="ExecutiveBoard",
    agents=[researcher, writer],
    verbose=True
)

# Execute a task
result = board_swarm.run("Create a market analysis report")
print(result)
```

### Custom Board Members

```python
from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
    BoardMember,
    BoardMemberRole
)

# Create custom board members
chairman = Agent(
    agent_name="CEO",
    agent_description="Chief Executive Officer",
    model_name="gpt-4o-mini"
)

cto = Agent(
    agent_name="CTO",
    agent_description="Chief Technology Officer", 
    model_name="gpt-4o-mini"
)

# Create board members with roles
board_members = [
    BoardMember(
        agent=chairman,
        role=BoardMemberRole.CHAIRMAN,
        voting_weight=2.0,
        expertise_areas=["leadership", "strategy"]
    ),
    BoardMember(
        agent=cto,
        role=BoardMemberRole.EXECUTIVE_DIRECTOR,
        voting_weight=1.5,
        expertise_areas=["technology", "innovation"]
    )
]

# Create swarm with custom board
swarm = BoardOfDirectorsSwarm(
    name="CustomBoard",
    board_members=board_members,
    agents=[researcher, writer]
)
```

## Configuration

### Environment Variables

The Board of Directors feature can be configured using environment variables:

```bash
# Enable the feature
export SWARMS_BOARD_FEATURE_ENABLED=true

# Configure board settings
export SWARMS_BOARD_DEFAULT_SIZE=4
export SWARMS_BOARD_DECISION_THRESHOLD=0.7
export SWARMS_BOARD_DEFAULT_MODEL=gpt-4o-mini
export SWARMS_BOARD_VERBOSE_LOGGING=true
export SWARMS_BOARD_ENABLE_VOTING=true
export SWARMS_BOARD_ENABLE_CONSENSUS=true
```

### Configuration File

Create a configuration file `swarms_board_config.yaml`:

```yaml
board_feature_enabled: true
default_board_size: 4
decision_threshold: 0.7
enable_voting: true
enable_consensus: true
default_board_model: gpt-4o-mini
verbose_logging: true
max_board_meeting_duration: 300
auto_fallback_to_director: true
custom_board_templates:
  executive:
    roles:
      - name: CEO
        weight: 2.0
        expertise: [leadership, strategy]
      - name: CFO
        weight: 1.5
        expertise: [finance, risk_management]
```

### Programmatic Configuration

```python
from swarms.config.board_config import (
    enable_board_feature,
    set_board_size,
    set_decision_threshold,
    enable_verbose_logging
)

# Enable and configure the feature
enable_board_feature("swarms_board_config.yaml")
set_board_size(4)
set_decision_threshold(0.7)
enable_verbose_logging("swarms_board_config.yaml")
```

## Board Member Roles

The Board of Directors supports various roles with different responsibilities:

### Available Roles

- **Chairman**: Primary leader responsible for board meetings and final decisions
- **Vice Chairman**: Secondary leader who supports the chairman
- **Secretary**: Responsible for documentation and meeting minutes
- **Treasurer**: Manages financial aspects and resource allocation
- **Member**: General board member with specific expertise
- **Executive Director**: Executive-level board member with operational authority

### Role Responsibilities

Each role has specific responsibilities and voting weights:

```python
# Chairman has highest voting weight
chairman = BoardMember(
    agent=ceo_agent,
    role=BoardMemberRole.CHAIRMAN,
    voting_weight=2.0,  # Higher weight for final decisions
    expertise_areas=["leadership", "strategy"]
)

# Executive directors have medium weight
cto = BoardMember(
    agent=cto_agent,
    role=BoardMemberRole.EXECUTIVE_DIRECTOR,
    voting_weight=1.5,
    expertise_areas=["technology", "innovation"]
)

# Regular members have standard weight
member = BoardMember(
    agent=member_agent,
    role=BoardMemberRole.MEMBER,
    voting_weight=1.0,
    expertise_areas=["operations"]
)
```

## Decision Making

The Board of Directors supports multiple decision-making mechanisms:

### Decision Types

- **Unanimous**: All board members agree on the decision
- **Majority**: More than 50% of votes are in favor
- **Consensus**: General agreement without formal voting
- **Chairman Decision**: Final decision made by the chairman

### Voting Configuration

```python
swarm = BoardOfDirectorsSwarm(
    agents=agents,
    decision_threshold=0.7,  # 70% majority required
    enable_voting=True,
    enable_consensus=True
)
```

## Task Execution

### Board Meeting Process

1. **Task Reception**: The board receives a task from the user
2. **Discussion**: Board members discuss the task and requirements
3. **Decision Making**: Board reaches consensus or majority decision
4. **Planning**: Board creates a detailed execution plan
5. **Task Distribution**: Board assigns specific tasks to worker agents
6. **Execution**: Worker agents execute tasks in parallel
7. **Feedback**: Board reviews results and provides feedback
8. **Iteration**: Process repeats if needed (up to max_loops)

### Task Distribution

The board distributes tasks using structured orders:

```python
# Example board order
order = BoardOrder(
    agent_name="Researcher",
    task="Conduct market research on AI trends",
    priority=1,  # High priority
    deadline="2024-01-15",
    assigned_by="Chairman"
)
```

### Parallel Execution

Tasks are executed in parallel for improved performance:

```python
swarm = BoardOfDirectorsSwarm(
    agents=agents,
    max_workers=4,  # Parallel execution with 4 workers
    verbose=True
)
```

## Advanced Features

### Async Support

The Board of Directors supports asynchronous operation:

```python
import asyncio

async def run_board_async():
    swarm = BoardOfDirectorsSwarm(agents=agents)
    result = await swarm.arun("Async task execution")
    return result

# Run async
result = asyncio.run(run_board_async())
```

### Board Member Management

Dynamic board member management:

```python
# Add a new board member
new_member = BoardMember(
    agent=new_agent,
    role=BoardMemberRole.MEMBER,
    voting_weight=1.0,
    expertise_areas=["marketing"]
)
swarm.add_board_member(new_member)

# Remove a board member
swarm.remove_board_member("MemberName")

# Get board member information
member = swarm.get_board_member("MemberName")
```

### Board Templates

Use predefined board templates:

```python
from swarms.config.board_config import get_board_config

config = get_board_config()
executive_template = config.get_default_board_template("executive")
advisory_template = config.get_default_board_template("advisory")
```

## Monitoring and Logging

### Verbose Logging

Enable detailed logging for debugging and monitoring:

```python
swarm = BoardOfDirectorsSwarm(
    agents=agents,
    verbose=True  # Enable detailed logging
)
```

### Board Summary

Get comprehensive board information:

```python
summary = swarm.get_board_summary()
print(f"Board Name: {summary['board_name']}")
print(f"Total Members: {summary['total_members']}")
print(f"Total Agents: {summary['total_agents']}")
print(f"Decision Threshold: {summary['decision_threshold']}")

for member in summary['members']:
    print(f"- {member['name']} ({member['role']}) - Weight: {member['voting_weight']}")
```

## Error Handling

The Board of Directors includes comprehensive error handling:

### Initialization Errors

```python
# Invalid configuration
try:
    swarm = BoardOfDirectorsSwarm(agents=[])  # No agents
except ValueError as e:
    print(f"Initialization error: {e}")

# Invalid parameters
try:
    swarm = BoardOfDirectorsSwarm(
        agents=agents,
        max_loops=0  # Invalid max_loops
    )
except ValueError as e:
    print(f"Parameter error: {e}")
```

### Runtime Error Handling

The system handles runtime errors gracefully:

- Agent execution failures are logged and reported
- Board meeting failures trigger fallback mechanisms
- Configuration errors are caught and reported
- Network and API errors are handled with retries

## Performance Optimization

### Parallel Execution

Optimize performance with parallel task execution:

```python
swarm = BoardOfDirectorsSwarm(
    agents=agents,
    max_workers=8,  # Increase for better parallel performance
    verbose=False   # Disable logging for production
)
```

### Memory Management

The system includes memory optimization features:

- Efficient conversation history management
- Lazy loading of board members
- Cached configuration and templates
- Automatic cleanup of completed tasks

## Best Practices

### Board Composition

1. **Balance Expertise**: Include board members with diverse expertise areas
2. **Appropriate Roles**: Assign roles based on responsibilities and authority
3. **Voting Weights**: Set voting weights based on decision-making authority
4. **Board Size**: Keep board size manageable (3-7 members recommended)

### Task Design

1. **Clear Objectives**: Provide clear and specific task descriptions
2. **Appropriate Scope**: Break large tasks into manageable components
3. **Resource Allocation**: Consider agent capabilities when assigning tasks
4. **Feedback Loops**: Use multiple loops for complex tasks requiring iteration

### Configuration

1. **Environment Variables**: Use environment variables for sensitive configuration
2. **Configuration Files**: Use YAML files for complex configurations
3. **Validation**: Validate configuration before deployment
4. **Monitoring**: Enable verbose logging for development and debugging

## Examples

### Complete Example

```python
from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
    BoardMember,
    BoardMemberRole
)
from swarms.structs.agent import Agent
from swarms.config.board_config import enable_board_feature

# Enable the feature
enable_board_feature()

# Create worker agents
researcher = Agent(
    agent_name="Market_Researcher",
    agent_description="Expert in market analysis and competitive intelligence",
    model_name="gpt-4o-mini"
)

analyst = Agent(
    agent_name="Data_Analyst",
    agent_description="Expert in data analysis and insights generation",
    model_name="gpt-4o-mini"
)

writer = Agent(
    agent_name="Report_Writer",
    agent_description="Expert in creating comprehensive reports",
    model_name="gpt-4o-mini"
)

# Create board members
ceo = Agent(
    agent_name="CEO",
    agent_description="Chief Executive Officer with strategic vision",
    model_name="gpt-4o-mini"
)

cto = Agent(
    agent_name="CTO",
    agent_description="Chief Technology Officer with technical expertise",
    model_name="gpt-4o-mini"
)

# Create board with custom members
board_members = [
    BoardMember(
        agent=ceo,
        role=BoardMemberRole.CHAIRMAN,
        voting_weight=2.0,
        expertise_areas=["leadership", "strategy", "business_development"]
    ),
    BoardMember(
        agent=cto,
        role=BoardMemberRole.EXECUTIVE_DIRECTOR,
        voting_weight=1.5,
        expertise_areas=["technology", "innovation", "product_development"]
    )
]

# Create the swarm
swarm = BoardOfDirectorsSwarm(
    name="ExecutiveBoard",
    description="Executive board for strategic decision making",
    board_members=board_members,
    agents=[researcher, analyst, writer],
    max_loops=2,
    verbose=True,
    decision_threshold=0.7,
    enable_voting=True,
    enable_consensus=True,
    max_workers=4
)

# Execute a complex task
task = """
Conduct a comprehensive analysis of our company's expansion into the European market.
This should include:
1. Market research and competitive landscape analysis
2. Financial feasibility and investment requirements
3. Technical infrastructure and operational requirements
4. Risk assessment and mitigation strategies

Provide a detailed report with recommendations for implementation.
"""

result = swarm.run(task)
print("Board of Directors execution completed!")
print(f"Result type: {type(result)}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the Board of Directors feature is enabled
2. **Configuration Errors**: Validate configuration files and environment variables
3. **API Errors**: Check API keys and network connectivity
4. **Performance Issues**: Adjust max_workers and disable verbose logging

### Debug Mode

Enable debug mode for troubleshooting:

```python
import logging
from swarms.utils.loguru_logger import initialize_logger

# Initialize detailed logging
logger = initialize_logger(log_folder="board_of_directors_debug")

# Create swarm with debug logging
swarm = BoardOfDirectorsSwarm(
    agents=agents,
    verbose=True  # Enable verbose logging
)
```

## API Reference

### BoardOfDirectorsSwarm

The main class for Board of Directors functionality.

#### Constructor

```python
BoardOfDirectorsSwarm(
    name: str = "BoardOfDirectorsSwarm",
    description: str = "Distributed task swarm with collective decision-making",
    board_members: Optional[List[BoardMember]] = None,
    agents: Optional[List[Union[Agent, Callable, Any]]] = None,
    max_loops: int = 1,
    output_type: OutputType = "dict-all-except-first",
    board_model_name: str = "gpt-4o-mini",
    verbose: bool = False,
    add_collaboration_prompt: bool = True,
    board_feedback_on: bool = True,
    decision_threshold: float = 0.6,
    enable_voting: bool = True,
    enable_consensus: bool = True,
    max_workers: Optional[int] = None,
    *args: Any,
    **kwargs: Any
)
```

#### Methods

- `run(task: str, img: Optional[str] = None, *args: Any, **kwargs: Any) -> Any`
- `arun(task: str, img: Optional[str] = None, *args: Any, **kwargs: Any) -> Any`
- `step(task: str, img: Optional[str] = None, *args: Any, **kwargs: Any) -> Any`
- `run_board_meeting(task: str, img: Optional[str] = None) -> BoardSpec`
- `add_board_member(board_member: BoardMember) -> None`
- `remove_board_member(agent_name: str) -> None`
- `get_board_member(agent_name: str) -> Optional[BoardMember]`
- `get_board_summary() -> Dict[str, Any]`

### BoardMember

Represents a member of the Board of Directors.

```python
BoardMember(
    agent: Agent,
    role: BoardMemberRole,
    voting_weight: float = 1.0,
    expertise_areas: List[str] = field(default_factory=list)
)
```

### BoardMemberRole

Enumeration of board member roles.

```python
class BoardMemberRole(str, Enum):
    CHAIRMAN = "chairman"
    VICE_CHAIRMAN = "vice_chairman"
    SECRETARY = "secretary"
    TREASURER = "treasurer"
    MEMBER = "member"
    EXECUTIVE_DIRECTOR = "executive_director"
```

### BoardDecisionType

Enumeration of decision types.

```python
class BoardDecisionType(str, Enum):
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    CONSENSUS = "consensus"
    CHAIRMAN_DECISION = "chairman_decision"
```

## Contributing

The Board of Directors feature follows the Swarms development guidelines:

1. **Code Quality**: Follow PEP 8 and use comprehensive type annotations
2. **Documentation**: Include detailed docstrings for all classes and methods
3. **Testing**: Write comprehensive tests for all functionality
4. **Performance**: Optimize for efficiency and scalability
5. **Error Handling**: Implement robust error handling and logging

## License

The Board of Directors feature is part of the Swarms Framework and follows the same licensing terms. 