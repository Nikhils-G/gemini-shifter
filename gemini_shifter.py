"""
Gemini Model Shifter - Resilient API client with automatic model fallback.

Features:
- Automatic model shifting on rate limits (429) and server errors (500, 503, 504)
- Exponential backoff retry logic
- Circuit breaker to temporarily disable failing models
- Environment-based configuration

Available Gemini Models (as of Jan 2026):
- gemini-2.5-flash: Fast & capable (recommended)
- gemini-2.5-flash-lite: Cheapest, high throughput
- gemini-2.5-pro: Most capable
- gemini-2.0-flash: Previous gen (retiring March 2026)
- gemini-2.0-flash-lite: Previous gen lite (retiring March 2026)
- gemini-3-flash-preview: Latest preview
- gemini-3-pro-preview: Latest preview (most advanced)
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

# Default model priority (if not set in .env)
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def load_config() -> dict:
    """Load configuration from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Check your .env file.")

    # Parse models from comma-separated string
    models_str = os.getenv("GEMINI_MODELS", "")
    models = [m.strip() for m in models_str.split(",") if m.strip()] or DEFAULT_MODELS

    return {
        "api_key": api_key,
        "models": models,
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "base_delay": float(os.getenv("BASE_DELAY_SECONDS", "1.0")),
        "max_delay": float(os.getenv("MAX_DELAY_SECONDS", "30.0")),
        "cb_max_failures": int(os.getenv("CIRCUIT_BREAKER_MAX_FAILURES", "3")),
        "cb_cooldown": int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "60")),
    }


@dataclass
class CircuitBreaker:
    """
    Tracks model failures and temporarily disables failing models.

    After max_failures consecutive failures, the model is disabled
    for cooldown_seconds before being tried again.
    """
    max_failures: int = 3
    cooldown_seconds: int = 60
    failure_counts: dict = field(default_factory=lambda: defaultdict(int))
    disabled_until: dict = field(default_factory=dict)

    def record_failure(self, model: str) -> None:
        """Record a failure for a model."""
        self.failure_counts[model] += 1
        if self.failure_counts[model] >= self.max_failures:
            self.disabled_until[model] = time.time() + self.cooldown_seconds
            logger.warning(f"Circuit OPEN for {model} - disabled for {self.cooldown_seconds}s")

    def record_success(self, model: str) -> None:
        """Reset failure count on success."""
        if model in self.failure_counts:
            self.failure_counts[model] = 0
        self.disabled_until.pop(model, None)

    def is_available(self, model: str) -> bool:
        """Check if a model is available (not circuit-broken)."""
        if model not in self.disabled_until:
            return True
        if time.time() > self.disabled_until[model]:
            # Cooldown expired, reset and allow
            self.disabled_until.pop(model)
            self.failure_counts[model] = 0
            logger.info(f"Circuit CLOSED for {model} - re-enabled")
            return True
        return False

    def get_status(self) -> dict:
        """Get current status of all tracked models."""
        return {
            "failure_counts": dict(self.failure_counts),
            "disabled_models": {
                model: round(until - time.time(), 1)
                for model, until in self.disabled_until.items()
                if until > time.time()
            }
        }


class GeminiShifter:
    """
    Resilient Gemini API client with automatic model fallback.

    Usage:
        shifter = GeminiShifter()
        response = shifter.generate("Your prompt here")
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the shifter.

        Args:
            config: Optional config dict. If None, loads from environment.
        """
        self.config = config or load_config()

        # Configure Gemini API
        genai.configure(api_key=self.config["api_key"])

        # Model settings
        self.models = self.config["models"]
        self.max_retries = self.config["max_retries"]
        self.base_delay = self.config["base_delay"]
        self.max_delay = self.config["max_delay"]

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            max_failures=self.config["cb_max_failures"],
            cooldown_seconds=self.config["cb_cooldown"]
        )

        # Track statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "model_usage": defaultdict(int),
            "errors_by_code": defaultdict(int),
        }

        logger.info(f"GeminiShifter initialized with models: {self.models}")

    def _extract_error_code(self, error: Exception) -> Optional[int]:
        """Extract HTTP status code from exception message."""
        error_str = str(error).lower()
        for code in RETRYABLE_ERRORS:
            if str(code) in error_str:
                return code
        # Check for common error patterns
        if "resource exhausted" in error_str or "quota" in error_str:
            return 429
        if "unavailable" in error_str:
            return 503
        return None

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate delay using exponential backoff with jitter."""
        delay = self.base_delay * (2 ** attempt)
        # Add small jitter to prevent thundering herd
        jitter = delay * 0.1 * (time.time() % 1)
        return min(delay + jitter, self.max_delay)

    def _try_model(
        self,
        model_name: str,
        prompt: str,
        generation_config: Optional[dict] = None,
        **kwargs
    ) -> tuple[bool, Optional[str], Optional[Exception]]:
        """
        Try to generate content with a specific model.

        Returns:
            Tuple of (success, response_text, error)
        """
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"[{model_name}] Attempt {attempt + 1}/{self.max_retries + 1}")

                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    **kwargs
                )

                # Success
                self.circuit_breaker.record_success(model_name)
                return True, response.text, None

            except Exception as e:
                error_code = self._extract_error_code(e)

                if error_code:
                    self.stats["errors_by_code"][error_code] += 1
                    logger.warning(f"[{model_name}] Error {error_code}: {str(e)[:100]}")
                    self.circuit_breaker.record_failure(model_name)

                    # Retry with backoff if attempts remain
                    if attempt < self.max_retries and error_code in RETRYABLE_ERRORS:
                        delay = self._calculate_backoff(attempt)
                        logger.info(f"[{model_name}] Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                else:
                    # Non-retryable error
                    logger.error(f"[{model_name}] Non-retryable error: {e}")

                return False, None, e

        return False, None, Exception(f"Max retries exceeded for {model_name}")

    def generate(
        self,
        prompt: str,
        generation_config: Optional[dict] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate content with automatic model fallback.

        Args:
            prompt: The text prompt to send
            generation_config: Optional generation settings (temperature, max_tokens, etc.)
            **kwargs: Additional arguments passed to generate_content

        Returns:
            Response text or None if all models fail
        """
        self.stats["total_requests"] += 1
        last_error = None

        for model_name in self.models:
            # Check circuit breaker
            if not self.circuit_breaker.is_available(model_name):
                logger.info(f"[{model_name}] SKIPPED (circuit open)")
                continue

            logger.info(f"[{model_name}] Trying model...")
            success, response, error = self._try_model(
                model_name, prompt, generation_config, **kwargs
            )

            if success:
                self.stats["successful_requests"] += 1
                self.stats["model_usage"][model_name] += 1
                logger.info(f"[{model_name}] SUCCESS")
                return response

            last_error = error
            logger.warning(f"[{model_name}] FAILED - shifting to next model...")

        # All models failed
        self.stats["failed_requests"] += 1
        logger.error(f"All models failed. Last error: {last_error}")
        return None

    def generate_with_metadata(
        self,
        prompt: str,
        generation_config: Optional[dict] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Generate content and return metadata about the request.

        Returns:
            Dict with 'response', 'model_used', 'success', 'error' keys
        """
        self.stats["total_requests"] += 1

        for model_name in self.models:
            if not self.circuit_breaker.is_available(model_name):
                continue

            success, response, error = self._try_model(
                model_name, prompt, generation_config, **kwargs
            )

            if success:
                self.stats["successful_requests"] += 1
                self.stats["model_usage"][model_name] += 1
                return {
                    "response": response,
                    "model_used": model_name,
                    "success": True,
                    "error": None
                }

        self.stats["failed_requests"] += 1
        return {
            "response": None,
            "model_used": None,
            "success": False,
            "error": str(error) if error else "All models failed"
        }

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            **self.stats,
            "model_usage": dict(self.stats["model_usage"]),
            "errors_by_code": dict(self.stats["errors_by_code"]),
            "circuit_breaker": self.circuit_breaker.get_status(),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "model_usage": defaultdict(int),
            "errors_by_code": defaultdict(int),
        }


# Convenience function for simple usage
def generate(prompt: str, **kwargs) -> Optional[str]:
    """
    Quick generate function using default configuration.

    Usage:
        from gemini_shifter import generate
        response = generate("Your prompt here")
    """
    shifter = GeminiShifter()
    return shifter.generate(prompt, **kwargs)


# --- Main / Demo ---
if __name__ == "__main__":
    print("=" * 60)
    print("Gemini Model Shifter - Demo")
    print("=" * 60)

    # Initialize shifter
    shifter = GeminiShifter()

    # Test prompt
    test_prompt = "Explain what an API rate limit is in one sentence."

    print(f"\nPrompt: {test_prompt}\n")
    print("-" * 60)

    # Generate with metadata
    result = shifter.generate_with_metadata(test_prompt)

    print("-" * 60)

    if result["success"]:
        print(f"\nModel Used: {result['model_used']}")
        print(f"Response: {result['response']}")
    else:
        print(f"\nFailed: {result['error']}")

    # Print stats
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    stats = shifter.get_stats()
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful_requests']}")
    print(f"Failed: {stats['failed_requests']}")
    print(f"Model Usage: {stats['model_usage']}")
    print(f"Errors by Code: {stats['errors_by_code']}")
