# Gemini Model Shifter - Technical Documentation

## Overview

The Gemini Model Shifter is a resilient API client that automatically handles model failures by shifting to backup models. It implements industry-standard patterns for fault tolerance in distributed systems.

---

## Architecture

```
+------------------+
|   User Request   |
+--------+---------+
         |
         v
+--------+---------+
|  GeminiShifter   |
|  (Main Client)   |
+--------+---------+
         |
         v
+--------+---------+     +-------------------+
|   Model Queue    |<--->|  Circuit Breaker  |
|  [model1, model2]|     |  (Health Tracker) |
+--------+---------+     +-------------------+
         |
         v
+--------+---------+
|   Try Model 1    |---> Success? Return response
+--------+---------+
         | Failure (429/500/503/504)
         v
+--------+---------+
|  Exponential     |
|  Backoff Retry   |
+--------+---------+
         | Max retries exceeded
         v
+--------+---------+
|   Try Model 2    |---> Continue chain...
+--------+---------+
```

---

## Core Components

### 1. Configuration Loader (`load_config`)

**Location:** `gemini_shifter.py:44-62`

Loads settings from environment variables with sensible defaults.

```python
config = {
    "api_key": "from .env",
    "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite", ...],
    "max_retries": 3,
    "base_delay": 1.0,
    "max_delay": 30.0,
    "cb_max_failures": 3,
    "cb_cooldown": 60,
}
```

**Why this is optimized:**
- Single source of truth for configuration
- Environment-based (12-factor app compliant)
- No hardcoded values in business logic
- Easy to modify without code changes

---

### 2. Circuit Breaker (`CircuitBreaker` class)

**Location:** `gemini_shifter.py:65-107`

Prevents repeated calls to a failing model, reducing wasted API calls and latency.

**States:**
```
CLOSED (Normal)  --->  OPEN (Blocked)  --->  HALF-OPEN (Testing)
     ^                      |                      |
     |                      | cooldown expires     |
     +----------------------+----------------------+
                    success resets
```

**How it works:**

| Event | Action |
|-------|--------|
| Request succeeds | Reset failure count to 0 |
| Request fails | Increment failure count |
| Failures >= max_failures | Open circuit (block model) |
| Cooldown expires | Close circuit (allow retry) |

**Code flow:**
```python
# On failure
circuit_breaker.record_failure("gemini-2.5-flash")
# After 3 failures, model is disabled for 60 seconds

# On success
circuit_breaker.record_success("gemini-2.5-flash")
# Resets failure count, model stays available

# Before trying a model
if circuit_breaker.is_available("gemini-2.5-flash"):
    # Proceed with request
else:
    # Skip to next model
```

**Why this is optimized:**
- Prevents "thundering herd" to a failing service
- Saves API quota by not retrying known-bad models
- Automatic recovery after cooldown
- O(1) lookup time using dictionaries

---

### 3. Exponential Backoff (`_calculate_backoff`)

**Location:** `gemini_shifter.py:139-143`

Calculates wait time between retries using exponential growth.

**Formula:**
```
delay = base_delay * (2 ^ attempt) + jitter
delay = min(delay, max_delay)
```

**Example progression:**
```
Attempt 0: 1.0s  (1 * 2^0)
Attempt 1: 2.0s  (1 * 2^1)
Attempt 2: 4.0s  (1 * 2^2)
Attempt 3: 8.0s  (1 * 2^3)
...capped at 30s max
```

**Jitter calculation:**
```python
jitter = delay * 0.1 * (time.time() % 1)
# Adds 0-10% random variation
```

**Why this is optimized:**
- Exponential growth gives server time to recover
- Jitter prevents synchronized retry storms from multiple clients
- Max cap prevents unreasonably long waits
- Industry standard pattern (used by AWS, Google, etc.)

---

### 4. Error Detection (`_extract_error_code`)

**Location:** `gemini_shifter.py:131-137`

Identifies retryable vs non-retryable errors.

**Retryable errors (shift to next model):**
| Code | Meaning | Action |
|------|---------|--------|
| 429 | Rate limit exceeded | Retry with backoff, then shift |
| 500 | Internal server error | Retry with backoff, then shift |
| 503 | Service unavailable | Retry with backoff, then shift |
| 504 | Gateway timeout | Retry with backoff, then shift |

**Non-retryable errors (fail immediately):**
| Code | Meaning | Action |
|------|---------|--------|
| 400 | Bad request | Fail (fix your prompt) |
| 401 | Unauthorized | Fail (fix your API key) |
| 403 | Forbidden | Fail (check permissions) |
| 404 | Not found | Fail (model doesn't exist) |

**Pattern matching:**
```python
# Catches various error message formats
if "resource exhausted" in error_str:  # Google's rate limit message
    return 429
if "unavailable" in error_str:  # Service down
    return 503
```

**Why this is optimized:**
- Only retries errors that can recover
- Fails fast on permanent errors (saves time)
- Handles multiple error message formats
- No wasted retries on bad requests

---

### 5. Model Shifting Logic (`generate`)

**Location:** `gemini_shifter.py:183-217`

The main request handler that orchestrates fallback.

**Flow:**
```
1. For each model in priority order:
   |
   +-> Is circuit open? Skip to next model
   |
   +-> Try request with retries
   |   |
   |   +-> Success? Return response
   |   |
   |   +-> Retryable error? Backoff and retry
   |   |
   |   +-> Max retries hit? Move to next model
   |
   +-> All models exhausted? Return None
```

**Priority order (configurable in .env):**
```
1. gemini-2.5-flash      (fast, capable)
2. gemini-2.5-flash-lite (fallback, cheaper)
3. gemini-2.0-flash      (older but stable)
4. gemini-2.0-flash-lite (last resort)
```

**Why this is optimized:**
- Priority order ensures best model is tried first
- Circuit breaker skips known-bad models instantly
- Each model gets fair retry attempts
- Graceful degradation (some response > no response)

---

## Optimization Techniques Used

### 1. Memory Efficiency

```python
# Using defaultdict instead of regular dict
self.failure_counts: dict = field(default_factory=lambda: defaultdict(int))
```
- No need to check if key exists
- Automatic initialization to 0
- Cleaner code, fewer conditionals

### 2. Time Complexity

| Operation | Complexity |
|-----------|------------|
| Check circuit breaker | O(1) |
| Record failure/success | O(1) |
| Calculate backoff | O(1) |
| Full request (worst case) | O(models * retries) |

### 3. Lazy Initialization

```python
# Model is only instantiated when needed
model = genai.GenerativeModel(model_name)
```
- No upfront cost for unused models
- Memory efficient

### 4. Statistics Tracking

```python
self.stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "model_usage": defaultdict(int),
    "errors_by_code": defaultdict(int),
}
```
- Useful for monitoring and debugging
- Helps identify which models are reliable
- Zero overhead when not accessed

---

## Configuration Reference

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Google AI API key |
| `GEMINI_MODELS` | gemini-2.5-flash,... | Comma-separated model list |
| `MAX_RETRIES` | 3 | Retries per model before shifting |
| `BASE_DELAY_SECONDS` | 1.0 | Initial backoff delay |
| `MAX_DELAY_SECONDS` | 30.0 | Maximum backoff delay |
| `CIRCUIT_BREAKER_MAX_FAILURES` | 3 | Failures before circuit opens |
| `CIRCUIT_BREAKER_COOLDOWN_SECONDS` | 60 | Seconds before circuit closes |

---

## Usage Examples

### Basic Usage

```python
from gemini_shifter import GeminiShifter

shifter = GeminiShifter()
response = shifter.generate("Explain quantum computing")
print(response)
```

### With Generation Config

```python
response = shifter.generate(
    "Write a haiku",
    generation_config={
        "temperature": 0.7,
        "max_output_tokens": 100,
    }
)
```

### Get Metadata (which model was used)

```python
result = shifter.generate_with_metadata("Hello")

print(result["response"])     # The actual response
print(result["model_used"])   # Which model succeeded
print(result["success"])      # True/False
```

### Check Statistics

```python
stats = shifter.get_stats()

print(f"Success rate: {stats['successful_requests']}/{stats['total_requests']}")
print(f"Most used model: {max(stats['model_usage'], key=stats['model_usage'].get)}")
print(f"Rate limits hit: {stats['errors_by_code'].get(429, 0)}")
```

### Quick One-Liner

```python
from gemini_shifter import generate

response = generate("Your prompt here")
```

---

## Comparison: With vs Without Shifter

### Without Shifter (Naive Approach)

```python
# Problem: Single point of failure
response = model.generate_content(prompt)
# If rate limited, entire application fails
```

**Issues:**
- No retry logic
- No fallback options
- Application crashes on 429/500 errors
- Poor user experience

### With Shifter

```python
# Solution: Resilient with automatic recovery
response = shifter.generate(prompt)
# Automatically handles failures, retries, and shifts models
```

**Benefits:**
- Automatic retry with backoff
- Multiple fallback models
- Circuit breaker prevents cascade failures
- Application stays responsive

---

## Performance Characteristics

### Best Case
```
Request -> Model 1 succeeds -> Return
Time: ~500ms (single API call)
```

### Typical Failure Recovery
```
Request -> Model 1 fails (429) -> Backoff 1s -> Retry -> Fails -> Model 2 succeeds
Time: ~2-3 seconds
```

### Worst Case
```
All models fail after all retries
Time: models * retries * avg_backoff = 4 * 3 * 5s = ~60 seconds
```

---

## When to Use This

**Use this when:**
- You are on a free/limited API tier
- Your application needs high availability
- You experience intermittent rate limits
- You want graceful degradation

**Not needed when:**
- You have a paid tier with high limits
- Single requests (no production load)
- You need guaranteed model consistency
- Latency is critical (fallback adds delay)

---

## Extending the Code

### Add a New Model

Edit `.env`:
```
GEMINI_MODELS=gemini-3-flash-preview,gemini-2.5-flash,gemini-2.5-flash-lite
```

### Add Custom Error Handling

```python
# In _extract_error_code method
if "custom_error_pattern" in error_str:
    return 429  # Treat as retryable
```

### Add Logging to File

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("gemini_shifter.log"),
        logging.StreamHandler()
    ]
)
```

---

## Summary

The Gemini Model Shifter implements three proven reliability patterns:

1. **Retry with Exponential Backoff** - Handles transient failures
2. **Circuit Breaker** - Prevents cascade failures
3. **Fallback Chain** - Ensures availability through redundancy

These patterns are used by major tech companies (Netflix, Amazon, Google) for building resilient systems. The implementation is optimized for:

- Minimal memory footprint
- O(1) health checks
- Zero overhead when healthy
- Configurable without code changes
