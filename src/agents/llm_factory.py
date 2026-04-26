"""LLM factory — Ollama, OpenAI, or Fireworks (Kimi K2.5) based on config."""
import logging
import os

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def build_llm() -> BaseChatModel:
    """Instantiate the configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    temperature = float(os.getenv("TEMPERATURE", "0.1"))

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        logger.info("Using Ollama model=%s", model)
        return ChatOllama(model=model, base_url=host, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI model=%s", model)
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)

    if provider == "fireworks":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY required when LLM_PROVIDER=fireworks")
        model = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2p5")
        base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        logger.info("Using Fireworks model=%s", model)
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider}. Must be 'ollama', 'openai', or 'fireworks'."
    )
