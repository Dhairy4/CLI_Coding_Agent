import atexit
from typing import Any
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from coding_agent.config import config

_POOL: ConnectionPool | None = None
_CHECKPOINTER: Any = None


def _cleanup_pool():
    global _POOL
    if _POOL is not None:
        try:
            _POOL.close()
        except Exception:
            pass


atexit.register(_cleanup_pool)


def get_checkpointer(force_memory: bool = False):
    """
    Provide persistent checkpointer for LangGraph state machine.
    Uses PostgresSaver with ConnectionPool when configured, falls back to MemorySaver.
    """
    global _POOL, _CHECKPOINTER

    if force_memory or not config.use_postgres_checkpointer:
        return MemorySaver()

    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    try:
        if _POOL is None:
            _POOL = ConnectionPool(
                config.postgres_uri,
                max_size=5,
                open=True,
                kwargs={"autocommit": True},
            )
        saver = PostgresSaver(_POOL)
        saver.setup()
        _CHECKPOINTER = saver
        return _CHECKPOINTER
    except Exception as e:
        print(f"[Warning] Failed to initialize PostgresSaver ({e}). Falling back to MemorySaver.")
        return MemorySaver()
