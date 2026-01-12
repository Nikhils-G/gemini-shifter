"""
Gemini Model Shifter - Resilient API client with automatic key and model fallback.

Features:
- Multiple API key support (from different GCP accounts)
- Automatic model shifting on rate limits (429) and server errors (500, 503, 504)
- Exponential backoff retry logic
- Circuit breaker for both keys and models
- Environment-based configuration

Fallback Order:
  Key1 -> [Model1, Model2, Model3, Model4]
  Key2 -> [Model1, Model2, Model3, Model4]
  Key3 -> [Model1, Model2, Model3, Model4]

Total attempts = Keys x Models x Retries
"""

import os
import time
import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Retryable HTTP error codes
RETRYABLE_ERRORS = {429, 500, 503, 504}

# Default model priority
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def load_config() -> dict:
    """Load configuration from environment variables."""

    # Support both single key (GEMINI_API_KEY) and multiple keys (GEMINI_API_KEYS)
    keys_str = os.getenv("GEMINI_API_KEYS", "")
    single_key = os.getenv("GEMINI_API_KEY", "")

    if keys_str:
        api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    elif single_key:
        api_keys = [single_key]
    else:
        raise ValueError("No API keys found. Set GEMINI_API_KEYS or GEMINI_API_KEY in .env")

    # Parse models
    models_str = os.getenv("GEMINI_MODELS", "")
    models = [m.strip() for m in models_str.split(",") if m.strip()] or DEFAULT_MODELS

    return {
        "api_keys": api_keys,
        "models": models,
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "base_delay": float(os.getenv("BASE_DELAY_SECONDS", "1.0")),
        "max_delay": float(os.getenv("MAX_DELAY_SECONDS", "30.0")),
        # Model circuit breaker
        "cb_max_failures": int(os.getenv("CIRCUIT_BREAKER_MAX_FAILURES", "3")),
        "cb_cooldown": int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "60")),
        # Key circuit breaker
        "key_cb_max_failures": int(os.getenv("KEY_CIRCUIT_BREAKER_MAX_FAILURES", "2")),
        "key_cb_cooldown": int(os.getenv("KEY_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "120")),
    }


def mask_key(key: str) -> str:
    """Mask API key for logging (show first 8 and last 4 chars)."""
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


@dataclass
class CircuitBreaker:
    """
    Tracks failures and temporarily disables failing resources.
    Used for both API keys and models.
    """
    name: str = "default"
    max_failures: int = 3
    cooldown_seconds: int = 60
    failure_counts: dict = field(default_factory=lambda: defaultdict(int))
    disabled_until: dict = field(default_factory=dict)

    def record_failure(self, resource: str) -> bool:
        """Record a failure. Returns True if circuit just opened."""
        self.failure_counts[resource] += 1
        if self.failure_counts[resource] >= self.max_failures:
            self.disabled_until[resource] = time.time() + self.cooldown_seconds
            logger.warning(
                f"[{self.name}] Circuit OPEN for {mask_key(resource) if 'key' in self.name.lower() else resource} "
                f"- disabled for {self.cooldown_seconds}s"
            )
            return True
        return False

    def record_success(self, resource: str) -> None:
        """Reset failure count on success."""
        if resource in self.failure_counts:
            self.failure_counts[resource] = 0
        self.disabled_until.pop(resource, None)

    def is_available(self, resource: str) -> bool:
        """Check if a resource is available."""
        if resource not in self.disabled_until:
            return True
        if time.time() > self.disabled_until[resource]:
            self.disabled_until.pop(resource)
            self.failure_counts[resource] = 0
            display_name = mask_key(resource) if 'key' in self.name.lower() else resource
            logger.info(f"[{self.name}] Circuit CLOSED for {display_name} - re-enabled")
            return True
        return False

    def get_status(self) -> dict:
        """Get current status."""
        return {
            "failure_counts": {
                (mask_key(k) if 'key' in self.name.lower() else k): v
                for k, v in self.failure_counts.items()
            },
            "disabled": {
                (mask_key(k) if 'key' in self.name.lower() else k): round(until - time.time(), 1)
                for k, until in self.disabled_until.items()
                if until > time.time()
            }
        }


class GeminiShifter:
    """
    Resilient Gemini API client with automatic key and model fallback.

    Fallback Strategy:
    1. Try each API key in order
    2. For each key, try each model in order
    3. For each model, retry with exponential backoff
    4. If all combinations fail, return None

    Usage:
        shifter = GeminiShifter()
        response = shifter.generate("Your prompt here")
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize the shifter."""
        self.config = config or load_config()

        # API Keys
        self.api_keys = self.config["api_keys"]
        logger.info(f"Loaded {len(self.api_keys)} API key(s)")

        # Models
        self.models = self.config["models"]
        logger.info(f"Models: {self.models}")

        # Retry settings
        self.max_retries = self.config["max_retries"]
        self.base_delay = self.config["base_delay"]
        self.max_delay = self.config["max_delay"]

        # Circuit breakers
        self.key_circuit_breaker = CircuitBreaker(
            name="Key-CB",
            max_failures=self.config["key_cb_max_failures"],
            cooldown_seconds=self.config["key_cb_cooldown"]
        )
        self.model_circuit_breaker = CircuitBreaker(
            name="Model-CB",
            max_failures=self.config["cb_max_failures"],
            cooldown_seconds=self.config["cb_cooldown"]
        )

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "key_usage": defaultdict(int),
            "model_usage": defaultdict(int),
            "errors_by_code": defaultdict(int),
            "key_shifts": 0,
            "model_shifts": 0,
        }

        # Calculate max attempts
        max_attempts = len(self.api_keys) * len(self.models) * (self.max_retries + 1)
        logger.info(f"Max attempts per request: {max_attempts} ({len(self.api_keys)} keys x {len(self.models)} models x {self.max_retries + 1} tries)")

    def _extract_error_code(self, error: Exception) -> Optional[int]:
        """Extract HTTP status code from exception message."""
        error_str = str(error).lower()
        for code in RETRYABLE_ERRORS:
            if str(code) in error_str:
                return code
        if "resource exhausted" in error_str or "quota" in error_str:
            return 429
        if "unavailable" in error_str:
            return 503
        return None

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate delay using exponential backoff with jitter."""
        delay = self.base_delay * (2 ** attempt)
        jitter = delay * 0.1 * (time.time() % 1)
        return min(delay + jitter, self.max_delay)

    def _try_with_key_and_model(
        self,
        api_key: str,
        model_name: str,
        prompt: str,
        generation_config: Optional[dict] = None,
        **kwargs
    ) -> tuple[bool, Optional[str], Optional[Exception]]:
        """
        Try to generate content with a specific key and model.
        Returns: (success, response_text, error)
        """
        # Configure API with this key
        genai.configure(api_key=api_key)
        key_display = mask_key(api_key)

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"[{key_display}] [{model_name}] Attempt {attempt + 1}/{self.max_retries + 1}")

                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    **kwargs
                )

                # Check if response has valid text
                if not response.parts:
                    raise Exception("Empty response - possibly blocked by safety filters")

                response_text = response.text

                # Success - reset circuit breakers
                self.key_circuit_breaker.record_success(api_key)
                self.model_circuit_breaker.record_success(model_name)
                return True, response_text, None

            except Exception as e:
                error_code = self._extract_error_code(e)

                if error_code:
                    self.stats["errors_by_code"][error_code] += 1
                    logger.warning(f"[{key_display}] [{model_name}] Error {error_code}: {str(e)[:80]}")

                    # Record failures
                    self.model_circuit_breaker.record_failure(model_name)

                    # For rate limits (429), also penalize the key
                    if error_code == 429:
                        self.key_circuit_breaker.record_failure(api_key)

                    # Retry with backoff if attempts remain
                    if attempt < self.max_retries and error_code in RETRYABLE_ERRORS:
                        delay = self._calculate_backoff(attempt)
                        logger.info(f"[{key_display}] [{model_name}] Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                else:
                    logger.error(f"[{key_display}] [{model_name}] Non-retryable: {e}")

                return False, None, e

        return False, None, Exception(f"Max retries exceeded for {model_name}")

    def generate(
        self,
        prompt: str,
        generation_config: Optional[dict] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate content with automatic key and model fallback.

        Fallback order:
        Key1 -> Model1, Model2, Model3...
        Key2 -> Model1, Model2, Model3...
        Key3 -> Model1, Model2, Model3...

        Returns: Response text or None if all combinations fail
        """
        self.stats["total_requests"] += 1
        last_error = None
        tried_keys = 0

        for api_key in self.api_keys:
            key_display = mask_key(api_key)

            # Check key circuit breaker
            if not self.key_circuit_breaker.is_available(api_key):
                logger.info(f"[{key_display}] SKIPPED (circuit open)")
                continue

            tried_keys += 1
            if tried_keys > 1:
                self.stats["key_shifts"] += 1
                logger.info(f"--- Shifting to key: {key_display} ---")

            tried_models = 0
            for model_name in self.models:
                # Check model circuit breaker
                if not self.model_circuit_breaker.is_available(model_name):
                    logger.info(f"[{key_display}] [{model_name}] SKIPPED (circuit open)")
                    continue

                tried_models += 1
                if tried_models > 1:
                    self.stats["model_shifts"] += 1

                success, response, error = self._try_with_key_and_model(
                    api_key, model_name, prompt, generation_config, **kwargs
                )

                if success:
                    self.stats["successful_requests"] += 1
                    self.stats["key_usage"][key_display] += 1
                    self.stats["model_usage"][model_name] += 1
                    logger.info(f"[{key_display}] [{model_name}] SUCCESS")
                    return response

                last_error = error
                logger.warning(f"[{key_display}] [{model_name}] FAILED - shifting...")

        # All combinations failed
        self.stats["failed_requests"] += 1
        logger.error(f"All keys and models failed. Last error: {last_error}")
        return None

    def generate_with_metadata(
        self,
        prompt: str,
        generation_config: Optional[dict] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Generate content and return metadata about the request.

        Returns dict with:
        - response: The generated text (or None)
        - key_used: Which API key succeeded (masked)
        - model_used: Which model succeeded
        - success: True/False
        - error: Error message if failed
        """
        self.stats["total_requests"] += 1
        last_error = None

        for api_key in self.api_keys:
            key_display = mask_key(api_key)

            if not self.key_circuit_breaker.is_available(api_key):
                continue

            for model_name in self.models:
                if not self.model_circuit_breaker.is_available(model_name):
                    continue

                success, response, error = self._try_with_key_and_model(
                    api_key, model_name, prompt, generation_config, **kwargs
                )

                if success:
                    self.stats["successful_requests"] += 1
                    self.stats["key_usage"][key_display] += 1
                    self.stats["model_usage"][model_name] += 1
                    return {
                        "response": response,
                        "key_used": key_display,
                        "model_used": model_name,
                        "success": True,
                        "error": None
                    }

                last_error = error

        self.stats["failed_requests"] += 1
        return {
            "response": None,
            "key_used": None,
            "model_used": None,
            "success": False,
            "error": str(last_error) if last_error else "All keys and models failed"
        }

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_requests": self.stats["total_requests"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": (
                f"{(self.stats['successful_requests'] / self.stats['total_requests'] * 100):.1f}%"
                if self.stats["total_requests"] > 0 else "N/A"
            ),
            "key_usage": dict(self.stats["key_usage"]),
            "model_usage": dict(self.stats["model_usage"]),
            "key_shifts": self.stats["key_shifts"],
            "model_shifts": self.stats["model_shifts"],
            "errors_by_code": dict(self.stats["errors_by_code"]),
            "key_circuit_breaker": self.key_circuit_breaker.get_status(),
            "model_circuit_breaker": self.model_circuit_breaker.get_status(),
        }

    def get_health(self) -> dict:
        """Get health status of all keys and models."""
        return {
            "keys": {
                mask_key(key): {
                    "available": self.key_circuit_breaker.is_available(key),
                    "failures": self.key_circuit_breaker.failure_counts.get(key, 0)
                }
                for key in self.api_keys
            },
            "models": {
                model: {
                    "available": self.model_circuit_breaker.is_available(model),
                    "failures": self.model_circuit_breaker.failure_counts.get(model, 0)
                }
                for model in self.models
            }
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "key_usage": defaultdict(int),
            "model_usage": defaultdict(int),
            "errors_by_code": defaultdict(int),
            "key_shifts": 0,
            "model_shifts": 0,
        }

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers (useful for testing)."""
        self.key_circuit_breaker = CircuitBreaker(
            name="Key-CB",
            max_failures=self.config["key_cb_max_failures"],
            cooldown_seconds=self.config["key_cb_cooldown"]
        )
        self.model_circuit_breaker = CircuitBreaker(
            name="Model-CB",
            max_failures=self.config["cb_max_failures"],
            cooldown_seconds=self.config["cb_cooldown"]
        )
        logger.info("All circuit breakers reset")


# Convenience function
def generate(prompt: str, **kwargs) -> Optional[str]:
    """Quick generate using default configuration."""
    shifter = GeminiShifter()
    return shifter.generate(prompt, **kwargs)


# --- Main / Demo ---
if __name__ == "__main__":
    print("=" * 70)
    print("Gemini Model Shifter - Multi-Key Demo")
    print("=" * 70)

    # Initialize
    shifter = GeminiShifter()

    # Test prompt
    test_prompt = "What is 2 + 2? Answer in one word."

    print(f"\nPrompt: {test_prompt}\n")
    print("-" * 70)

    # Generate with metadata
    result = shifter.generate_with_metadata(test_prompt)

    print("-" * 70)

    if result["success"]:
        print(f"\nKey Used:   {result['key_used']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Response:   {result['response']}")
    else:
        print(f"\nFailed: {result['error']}")

    # Health check
    print("\n" + "=" * 70)
    print("Health Status:")
    print("=" * 70)
    health = shifter.get_health()

    print("\nAPI Keys:")
    for key, status in health["keys"].items():
        status_str = "OK" if status["available"] else "DISABLED"
        print(f"  {key}: {status_str} (failures: {status['failures']})")

    print("\nModels:")
    for model, status in health["models"].items():
        status_str = "OK" if status["available"] else "DISABLED"
        print(f"  {model}: {status_str} (failures: {status['failures']})")

    # Stats
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    stats = shifter.get_stats()
    print(f"Total Requests:  {stats['total_requests']}")
    print(f"Successful:      {stats['successful_requests']}")
    print(f"Failed:          {stats['failed_requests']}")
    print(f"Success Rate:    {stats['success_rate']}")
    print(f"Key Shifts:      {stats['key_shifts']}")
    print(f"Model Shifts:    {stats['model_shifts']}")
