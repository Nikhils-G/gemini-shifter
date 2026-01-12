# Gemini Model Shifter

Resilient Gemini API client with automatic key and model fallback.

## Features

- Multiple API keys support (from different GCP accounts)
- Automatic model shifting on errors (429, 500, 503, 504)
- Exponential backoff retry
- Circuit breaker for keys and models

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from gemini_shifter import GeminiShifter

shifter = GeminiShifter()
response = shifter.generate("Your prompt here")
```

## Configuration (.env)

```
# API Keys (comma-separated)
GEMINI_API_KEYS=key1,key2,key3

# Models (priority order)
GEMINI_MODELS=gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash,gemini-2.0-flash-lite

# Retry settings
MAX_RETRIES=3
BASE_DELAY_SECONDS=1.0
MAX_DELAY_SECONDS=30.0

# Circuit breaker
CIRCUIT_BREAKER_MAX_FAILURES=3
CIRCUIT_BREAKER_COOLDOWN_SECONDS=60
KEY_CIRCUIT_BREAKER_MAX_FAILURES=2
KEY_CIRCUIT_BREAKER_COOLDOWN_SECONDS=120
```

## How It Works

```
Key1 -> [Model1 -> Model2 -> Model3 -> Model4]
  |
  v (if all models fail)
Key2 -> [Model1 -> Model2 -> Model3 -> Model4]
  |
  v
Key3 -> ...
```

## Timing (Worst Case)

| Config | Retries | Total Attempts | Time |
|--------|---------|----------------|------|
| Conservative | 3 | 96 | ~7.5 min |
| Balanced | 2 | 72 | ~4.2 min |
| Fast | 1 | 48 | ~2.6 min |
| Aggressive | 0 | 24 | ~1.2 min |

*Based on 6 keys x 4 models*

## Usage Examples

```python
# Basic
response = shifter.generate("Hello")

# With config
response = shifter.generate("Hello", generation_config={"temperature": 0.7})

# With metadata
result = shifter.generate_with_metadata("Hello")
print(result["model_used"])
print(result["key_used"])

# Health check
health = shifter.get_health()

# Statistics
stats = shifter.get_stats()
```

## Error Handling

| Code | Action |
|------|--------|
| 429 | Retry + shift model + penalize key |
| 500, 503, 504 | Retry + shift model |
| 400, 401, 403 | Fail immediately |


Sincearly,
Nikhil Sukthe