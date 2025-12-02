# Autonomous Microbiz Hardening Guide

## Overview

This guide documents the transition from simulated/advisory mode to a fully autonomous micro-business system using Swarms.

## Autonomy Modes

### Advisory Mode (Default)
- Board makes recommendations only
- No automatic actions taken
- Human approval required for all operations
- Safe for initial testing

**Usage:**
```python
from business.integration import BusinessIntegration

biz = BusinessIntegration()
# Mode is "advisory" by default
recommendations = biz.get_board_recommendations()
# Review and manually execute recommendations
```

### Semi-Auto Mode
- Free content (GitHub releases, Substack posts) published automatically
- Paid products (Gumroad) require manual approval
- Scaling and pricing decisions are advisory only

**Configuration:**
```json
{
  "autonomy": {
    "mode": "semi_auto",
    "kill_switch": false
  }
}
```

**Usage:**
```python
biz.config["autonomy"]["mode"] = "semi_auto"
result = biz.process_collected_data("auctions", data, metadata)
# GitHub/Substack posts are published automatically
# Gumroad listings are created but not published
```

### Full-Auto Mode
- All operations execute automatically
- Board decisions are applied immediately
- Pricing and scaling changes are automatic
- **Use with caution** - requires proper monitoring

**Configuration:**
```json
{
  "autonomy": {
    "mode": "full_auto",
    "kill_switch": false
  }
}
```

## Kill Switch

Emergency stop mechanism that forces system back to advisory mode.

### Config-Based Kill Switch
```json
{
  "autonomy": {
    "kill_switch": true
  }
}
```

### Environment Variable Kill Switch
```bash
export MICROBIZ_KILL_SWITCH=true
```

When activated, the system will:
- Force `mode` to `"advisory"`
- Log a warning
- Prevent all automatic actions

## Integration Setup

### GitHub Integration
```bash
export GITHUB_TOKEN="your_token"
export GITHUB_REPO="owner/repo"
```

### Substack Integration
```bash
export SUBSTACK_API_KEY="your_key"
export SUBSTACK_PUBLICATION_ID="your_id"
```

### Gumroad Integration
```bash
export GUMROAD_API_KEY="your_key"
```

## Board of Directors

The Board makes strategic decisions based on:
- Worker performance metrics
- Financial status
- Market opportunities

### Status Report
```python
status_report = biz.build_board_status_report()
# Contains:
# - worker_performance: Success rates, quality, error counts
# - financial_status: Revenue, costs, profit margins
# - market_opportunities: High-value data sources
```

### Scaling Decisions
```python
decision = biz.make_scaling_decision("auctions", performance_metrics)
# Returns: {"action": "scale_up", "new_frequency": "01:00", "reason": "..."}
```

### Pricing Decisions
```python
decision = biz.make_pricing_decision("premium", performance_metrics)
# Returns: {"action": "increase", "new_price": 59.99, "reason": "..."}
```

## Worker Hardening

All workers now include:
- **Rate Limiting**: Configurable delays, user agent rotation
- **Validation**: Pydantic schema validation, quality scoring
- **Error Handling**: Structured error management, retry logic
- **Adaptive Parsing**: LLM-based parsing for unknown structures
- **Metrics Collection**: Performance snapshots for Board consumption

## Examples

See `examples/` directory:
- `run_advisory.py`: Advisory mode example
- `run_semi_auto.py`: Semi-auto mode example

## Safety Guidelines

1. **Start in Advisory Mode**: Always begin with advisory mode
2. **Monitor First**: Run for several cycles and review Board recommendations
3. **Gradual Escalation**: Move to semi-auto only after verifying behavior
4. **Full-Auto Caution**: Use full-auto only with proper monitoring and caps
5. **Kill Switch Ready**: Keep kill switch accessible for emergencies

## Monitoring

Key metrics to monitor:
- Worker success rates
- Data quality scores
- Error counts and types
- Financial performance
- Board decision frequency
- Integration API call success rates

## Troubleshooting

### Board Not Making Decisions
- Check if Board is enabled in `config/business_config.json`
- Verify OpenAI API key is set
- Review Board logs for errors

### Integrations Not Working
- Verify API keys are set in environment
- Check integration `enabled` flags in config
- Review autonomy mode (must be semi_auto or full_auto for auto-publishing)

### Scaling Not Applied
- Check autonomy mode (must be semi_auto or full_auto)
- Verify supervisor is reading scaling decisions
- Review supervisor logs for scaling plan updates

