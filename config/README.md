# Configuration

Centralized configuration management for the project.

## Files

- `config.json` - Runtime configuration (DO NOT version control with secrets)
- `config.example.json` - Configuration template (version control this)

## Usage

Load configuration in your code:

```python
import json
from pathlib import Path

config_path = Path(__file__).parent.parent / "config" / "config.json"
with open(config_path) as f:
    config = json.load(f)
```

## Best Practices

- Never commit `config.json` with real credentials
- Always version control `config.example.json` as a template
- Use environment variables for sensitive data
- Add new config options to both files when making changes
- Document config options in this file
