#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorator for marking agent methods as task handlers."""


def task(fn=None, *, name: str | None = None, parallel: bool = False):
    """Mark an agent method as a task handler.

    Decorated methods are automatically collected by ``BaseAgent`` at
    initialization and dispatched when matching task requests arrive.

    Can be used with or without arguments::

        @task
        async def on_task_request(self, message):
            ...

        @task(parallel=True)
        async def on_task_request(self, message):
            ...

        @task(name="research")
        async def on_research(self, message):
            ...

        @task(name="research", parallel=True)
        async def on_research(self, message):
            ...

    Args:
        fn: The function to decorate (when used without arguments).
        name: Optional task name to match. When set, this handler only
            receives requests with a matching name. When None, handles
            all unnamed requests (or requests with no matching named
            handler).
        parallel: When True, each request runs in a separate asyncio
            task for concurrent execution. Defaults to False.
    """

    def decorator(fn):
        fn.is_task_handler = True
        fn.task_name = name
        fn.task_parallel = parallel
        return fn

    if fn is not None:
        return decorator(fn)
    return decorator


def _collect_task_handlers(obj) -> dict:
    """Collect all ``@task`` decorated bound methods from an object.

    Returns a dict mapping task name (or None for the default handler)
    to a tuple of (method, is_async).

    Raises:
        ValueError: If two handlers share the same task name.
    """
    seen: set[str] = set()
    handlers: dict[str | None, tuple] = {}
    for cls in type(obj).__mro__:
        for attr_name, val in cls.__dict__.items():
            if attr_name in seen:
                continue
            seen.add(attr_name)
            if callable(val) and getattr(val, "is_task_handler", False):
                task_name = val.task_name
                if task_name in handlers:
                    existing = handlers[task_name][0].__name__
                    label = f"'{task_name}'" if task_name else "default (unnamed)"
                    raise ValueError(
                        f"Duplicate @task handler for {label}: "
                        f"'{attr_name}' conflicts with '{existing}'"
                    )
                bound = getattr(obj, attr_name)
                handlers[task_name] = (bound, val.task_parallel)
    return handlers
