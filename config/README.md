# Configuration

Centralized configuration management for the project.

## Files

- `bot_config.json` - All runtime configuration (trading, risk, MT5 bridge, etc.)

## Usage

Load configuration in your code:

```python
import json
from pathlib import Path

config_path = Path(__file__).parent.parent / "config" / "bot_config.json"
with open(config_path) as f:
    config = json.load(f)
```

## Best Practices

- Use environment variables for sensitive data
- Document config options in this file
