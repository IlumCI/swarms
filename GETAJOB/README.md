# GETAJOB - Autonomous Job-Seeking Agent System

A sophisticated multi-agent system that autonomously searches for freelancing jobs, applies to opportunities, completes work, and generates real income.

## Architecture

The system uses a hybrid architecture combining:

- **AERASIGMA**: Strategic planning and knowledge base management
- **CR-CA Agent**: Causal reasoning with `max_loops="auto"` for continuous analysis
- **Chain of Thought (CoT)**: Step-by-step job searching
- **Tree of Thoughts (ToT)**: Multiple proposal variations
- **Graph of Thoughts (GoT)**: Complex work task decomposition

## Features

- **Autonomous Job Searching**: Searches multiple freelancing platforms (Upwork, Fiverr)
- **Intelligent Application Writing**: Uses ToT to generate multiple proposal variations
- **Causal Market Analysis**: CR-CA agent predicts application success probability
- **Work Execution**: GoT agent decomposes and completes complex tasks
- **Income Tracking**: Comprehensive tracking and reporting system
- **Human-like Behavior**: Natural delays, language variations, and learning

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create `.env` file):
```env
MODEL_NAME=gpt-4o
ORCHESTRATOR_MODEL=gpt-4o
ANALYST_MODEL=gpt-4o
UPWORK_USERNAME=your_username
UPWORK_PASSWORD=your_password
FIVERR_USERNAME=your_username
FIVERR_PASSWORD=your_password
OPENAI_API_KEY=your_api_key
```

## Usage

Run the main executable:

```bash
python GETAJOB/getajob.py
```

The system will:
1. Search for jobs matching your target skills
2. Analyze opportunities using causal reasoning
3. Write and submit personalized proposals
4. Track applications and income
5. Generate reports for the "corporation"

## Configuration

Edit the configuration section in `getajob.py`:

- `TARGET_SKILLS`: Skills to search for
- `MIN_JOB_BUDGET`: Minimum acceptable job budget
- `MAX_APPLICATIONS_PER_DAY`: Rate limiting
- `MIN_DELAY` / `MAX_DELAY`: Human behavior timing
- `cycle_interval`: Time between search cycles (in seconds)

## File Structure

```
GETAJOB/
├── __init__.py          # Package initialization
├── getajob.py           # Main executable (all functionality)
├── requirements.txt     # Dependencies
├── README.md           # This file
└── data/               # Auto-created data directory
    ├── job_memory.json  # Application history
    └── income.db        # Income tracking database
```

## Components

### JobSeekerOrchestrator
Main coordinator using AERASIGMA for strategic planning.

### JobMarketAnalyst
CR-CA agent with `max_loops="auto"` for causal analysis of job market trends.

### JobHunter
CoT agent for systematic job searching across platforms.

### ApplicationWriter
ToT agent for generating multiple proposal variations and selecting the best.

### WorkExecutor
GoT agent for decomposing and executing complex work tasks.

### IncomeManager
Tracks all income sources and generates reports.

## Notes

- Browser automation requires `browser-use` and `langchain-openai`
- Web search fallback uses DuckDuckGo if browser automation fails
- All data is stored locally in the `data/` directory
- The system learns from past applications to improve success rates

## License

Part of the Swarms framework.

