import asyncio
from functools import wraps
from typing import Callable, Coroutine, Any

global_task_list = []


def fire_and_forget(func: Callable[..., Coroutine[Any, Any, Any]]):
    @wraps(func)
    def wrapper(*args, **kwargs):
        task = asyncio.create_task(func(*args, **kwargs))

        # Optional: Store the task somewhere if you need to keep track of it
        global_task_list.append(task)

        # Optional: Add error handling
        task.add_done_callback(lambda t: t.exception())

        return task

    return wrapper
