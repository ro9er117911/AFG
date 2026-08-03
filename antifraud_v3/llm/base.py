from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMProviderError(Exception):
    """Provider-agnostic wrapper — callers in reasoning/ catch this, never an SDK-specific error."""


class LLMProvider(ABC):
    """Single-method interface between the reasoning engine (reasoning/) and whichever
    model actually answers it. See REWRITE_PLAN.md §3 for why this is deliberately this thin:
    the reasoning engine never imports a provider SDK directly, only this interface.
    """

    @abstractmethod
    def structured_complete(
        self,
        system: str,
        user_content: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        """Send a single structured-output request; return a validated instance of `schema`."""
        raise NotImplementedError
