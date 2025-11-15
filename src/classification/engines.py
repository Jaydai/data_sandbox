"""Inference engines powering the classifier service."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

try:  # pragma: no cover - optional dependency in some runtimes
    import ollama  # type: ignore
except ModuleNotFoundError:  # defer error until the engine is instantiated
    ollama = None  # type: ignore

logger = logging.getLogger(__name__)


class ClassificationEngine(ABC):
    """Minimal interface every inference backend must implement."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Return the raw completion string."""


class OllamaEngine(ClassificationEngine):
    """Run prompts against a local Ollama model."""

    def __init__(self, model: str = "qwen2.5:14b"):
        if ollama is None:
            raise ImportError(
                "Le module 'ollama' est requis pour OllamaEngine. "
                "Installez-le (pip install ollama) ou utilisez HFRouterEngine."
            )
        super().__init__(model_name=model)
        self._ensure_model_ready()

    def _ensure_model_ready(self) -> None:
        """Warm up model and pull it if needed."""
        test_prompt = {
            "role": "user",
            "content": "ping",
        }
        try:
            ollama.chat(
                model=self.model_name,
                messages=[test_prompt],
                options={"num_predict": 5},
            )
            logger.info("✅ Ollama model %s prêt", self.model_name)
        except ollama.ResponseError as exc:
            if "not found" in str(exc).lower():
                logger.info("⬇️ Téléchargement du modèle %s", self.model_name)
                ollama.pull(self.model_name)
            else:
                raise

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]


class HFRouterEngine(ClassificationEngine):
    """Use Hugging Face Router through the OpenAI compatible API."""

    AVAILABLE_MODELS = {
        "gemma-9b": "google/gemma-2-9b-it:nebius",
        "llama-8b": "meta-llama/llama-3.1-8b-instruct:nebius",
        "mistral-7b": "mistralai/mistral-7b-instruct-v0.3:nebius",
        "qwen-7b": "qwen/qwen-2.5-7b-instruct:nebius",
    }

    def __init__(self, model: str = "gemma-9b", api_key: Optional[str] = None):
        load_dotenv()
        resolved_model = self.AVAILABLE_MODELS.get(model, model)
        super().__init__(model_name=resolved_model)

        token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        if not token:
            raise ValueError("HF_TOKEN ou HUGGINGFACE_API_KEY doit être défini dans l'environnement")

        self._client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not completion.choices:
            return ""
        return completion.choices[0].message.content.strip()
