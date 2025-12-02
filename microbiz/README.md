# Data Swarm - Monochrome Multi-Task Data Collection System

A minimal, self-scheduling data collection swarm system using the Swarms framework. Each worker is a Swarms Agent that collects and structures data from various sources.

## Features

### Data Collection
- **Self-Scheduling**: Time-based scheduling with no cron dependencies
- **Swarms Integration**: Each worker is a Swarms Agent with tools
- **Real Data Sources**: Configurable real URLs with fallback chains
- **Hybrid Parsing**: LLM-based adaptive parsing + structured parsers
- **Rate Limiting**: Respectful scraping with delays, user agent rotation, robots.txt respect
- **Data Validation**: Pydantic schema validation and quality scoring
- **Robust Error Handling**: Site change detection, graceful degradation, error categorization
- **Cloud Shell Compatible**: Runs continuously in Google Cloud Shell

### Business Operations
- **Board of Directors**: Collective decision-making through voting and consensus
- **Revenue Generation**: Multiple pricing models (subscription, one-time, tiered, API)
- **Financial Management**: Revenue/cost tracking, P&L calculations, budget allocation
- **Operations Automation**: Sales (Gumroad), Marketing (Substack), Samples (GitHub)
- **Autonomous Behavior**: Self-optimization, automatic scaling, market opportunity detection
- **Strategic Decision-Making**: Board-driven decisions on pricing, scaling, and pivoting

### Technical
- **GCS Sync**: Optional Google Cloud Storage synchronization
- **LLM Summarization**: Optional summarization using Swarms Agents or llama.cpp
- **Memory System Support**: Optional long-term memory with ChromaDB, FAISS, or Qdrant
- **AgentRearrange Ready**: Optional multi-agent workflow orchestration

## Architecture

```
microbiz/
├── supervisor.py          # Time-based scheduler with Board integration
├── config.json            # Main configuration
├── config/
│   ├── sources.json       # Real URLs and source configurations
│   └── business_config.json  # Business logic configuration
├── requirements.txt       # Dependencies
├── workers/               # Swarms Agent workers
│   ├── auctions.py        # Production-ready with rate limiting & validation
│   ├── tenders.py
│   ├── businesses.py
│   ├── jobs.py
│   ├── osint.py
│   └── realestate.py
├── modules/               # Shared utilities
│   ├── tools.py           # Custom Swarms tools
│   ├── parser.py          # JSON parsing
│   ├── summarizer.py      # LLM summarization
│   ├── storage.py         # Local + GCS storage
│   ├── memory.py          # Memory system initialization
│   ├── rate_limiter.py    # Rate limiting & anti-scraping
│   ├── validator.py       # Data validation & quality scoring
│   ├── adaptive_parser.py # LLM-based adaptive parsing
│   └── error_handler.py   # Robust error handling
├── business/              # Business logic & Board of Directors
│   ├── board.py           # BoardOfDirectorsSwarm setup
│   ├── revenue.py         # Revenue generation strategies
│   ├── operations.py      # Business operations automation
│   ├── finance.py         # Financial management (Treasurer)
│   ├── autonomy.py        # Autonomous behavior engine
│   └── integration.py    # Business integration coordinator
├── integrations/          # External service integrations
│   ├── gumroad.py         # Gumroad API
│   ├── substack.py        # Substack API
│   └── github.py          # GitHub API
└── output/                # Generated data files
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Set your LLM API keys (required for Swarms Agents):

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL_NAME="gpt-4o-mini"  # Optional, defaults to gpt-4o-mini
```

For llama.cpp fallback (optional):

```bash
export LLAMA_MODEL_PATH="/path/to/model.gguf"
export LLAMA_CPP_BINARY="llama-cli"  # or path to llama.cpp binary
```

### 3. Configure Configuration Files

**config.json** - Main configuration:
- **Schedule**: Times for each worker (HH:MM format)
- **Storage**: GCS bucket name (optional)
- **LLM**: Enable/disable summarization
- **Memory**: Optional memory system configuration
- **Supervisor**: Check interval, retry settings

**config/sources.json** - Real data sources:
- Configure actual URLs for each worker type
- Set priorities and fallback chains
- Enable/disable specific sources

**config/business_config.json** - Business configuration:
- **Board**: Enable Board of Directors, voting weights, decision thresholds
- **Revenue**: Pricing models and strategies
- **Operations**: Gumroad, Substack, GitHub API keys
- **Autonomy**: Auto-scaling and optimization settings

Example:

```json
{
  "schedule": {
    "auctions": "02:00",
    "tenders": "02:20",
    "businesses": "02:40",
    "jobs": "03:00",
    "osint": "03:20",
    "realestate": "03:40"
  },
  "storage": {
    "gcs_bucket": "data-swarm-bucket"
  },
  "llm": {
    "enabled": false,
    "prefer_swarms": true
  },
  "memory": {
    "enabled": false,
    "type": "chromadb",
    "output_dir": "data_swarm_memory"
  },
  "supervisor": {
    "use_agent_rearrange": false
  }
}
```

## Deployment

### Google Cloud Shell

1. Clone or upload the `microbiz` folder to Cloud Shell
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables
4. Start in tmux session:

```bash
tmux new -s swarm
cd microbiz
python3 supervisor.py
```

5. Detach: `Ctrl+B` then `D`

6. Reattach: `tmux attach -t swarm`

### Local Development

```bash
cd microbiz
python3 supervisor.py
```

## Usage

### Running Individual Workers

Each worker can be run independently:

```bash
python3 workers/auctions.py
python3 workers/tenders.py
# etc.
```

### Supervisor Mode

The supervisor continuously monitors the schedule and runs workers at their scheduled times:

```bash
python3 supervisor.py
```

## Output Format

All workers output standardized JSON:

```json
{
  "generated_at": "2025-01-15T02:00:00Z",
  "source": "auctions",
  "data": [...],
  "summary": "optional summary text"
}
```

Output files are saved to `output/<worker-name>/<timestamp>.json`

## Workers

### auctions.py
Scrapes auction sites and extracts: title, price, bids, end_time

### tenders.py
Parses tender portals and extracts: deadline, authority, budget, category

### businesses.py
Monitors business registries for: new registrations, deregistrations

### jobs.py
Extracts job postings with: tech stack, salary, location

### osint.py
Collects OSINT data: server headers, tech stack, metadata

### realestate.py
Scrapes listings and computes: price per square meter, district metrics

## Customization

### Adding Custom Tools

Edit `modules/tools.py` to add new tool functions. Tools are automatically available to all workers.

### Modifying Workers

Each worker in `workers/` can be customized:
- Change system prompts
- Add/remove tools
- Modify task descriptions
- Adjust parsing logic

### Changing Schedule

Edit `config.json` schedule section. Times are in HH:MM format (24-hour).

### Production Features

**Rate Limiting & Anti-Scraping**:
- Automatic delays between requests
- User agent rotation
- robots.txt respect
- Exponential backoff on rate limits

**Data Validation**:
- Pydantic schema validation
- Quality scoring (0.0-1.0)
- Duplicate detection
- Quality filtering

**Adaptive Parsing**:
- LLM-based extraction for unknown sites
- Automatic parser selection
- Site structure change detection
- Graceful degradation

### Board of Directors Configuration

To enable Board of Directors governance:

1. Edit `config/business_config.json`:
```json
{
  "board": {
    "enabled": true,
    "decision_threshold": 0.6,
    "enable_voting": true,
    "enable_consensus": true
  }
}
```

2. Board members:
   - **Chairman**: Strategic leadership (weight: 1.5)
   - **Treasurer**: Financial oversight (weight: 1.0)
   - **Secretary**: Documentation (weight: 1.0)
   - **Executive Director**: Operations (weight: 1.2)
   - **Vice Chairman**: Day-to-day management (weight: 1.2)

3. Board makes decisions on:
   - Worker scaling (up/down)
   - Pricing strategies
   - Budget allocation
   - Strategic pivoting

### Memory System Configuration

See previous section for ChromaDB, FAISS, and Qdrant setup.

### Revenue Generation

Configure revenue strategies in `config/business_config.json`:
- Subscription models (weekly/monthly/yearly)
- One-time dataset sales
- Tiered pricing (free/basic/premium/enterprise)
- API access pricing

## Troubleshooting

### Workers Not Running

- Check that current time matches scheduled time (within 30-second check interval)
- Verify environment variables are set
- Check logs for errors

### GCS Sync Fails

- Ensure `gsutil` is installed and configured
- Verify bucket name is correct
- Check GCS permissions

### LLM Errors

- Verify API keys are set correctly
- Check model name is valid
- Review summarizer logs for details

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path includes microbiz directory
- Verify Swarms framework is installed

## License

Part of the Swarms framework ecosystem.

