try:
    from .env.exalted_environment import ExaltedEnv
except ImportError:  # pragma: no cover - fallback for direct execution
    from env.exalted_environment import ExaltedEnv
