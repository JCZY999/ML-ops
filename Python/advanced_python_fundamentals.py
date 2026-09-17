"""Advanced Python fundamentals through practical MLOps-style examples.

Use these tools to write pipelines that are readable, memory-efficient,
reusable, and easy to instrument. Run this file to see each example.
"""

from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from functools import lru_cache, partial, reduce, wraps
from itertools import chain, combinations, islice
from time import perf_counter


# Comprehensions: transform/filter records compactly.
metrics = {"loss": 0.21, "accuracy": 0.94, "latency_ms": 37}
good_metrics = {name: value for name, value in metrics.items() if value > 0.5}
unique_stages = {stage for stage in ["train", "test", "train", "deploy"]}
squared_ids = [item_id**2 for item_id in range(5)]


# Generators and yield: stream large datasets without loading everything.
def batches(items, size):
    """Yield one fixed-size batch at a time."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


# Iterators: objects implementing __iter__ and __next__ control traversal.
class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value


# Decorators: add logging/timing without changing business logic.
def timed(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        started = perf_counter()
        result = function(*args, **kwargs)
        elapsed = perf_counter() - started
        print(f"{function.__name__} took {elapsed:.6f}s")
        return result

    return wrapper


# Closures: retain configuration in an inner function.
def threshold_checker(limit):
    def is_acceptable(value):
        return value <= limit

    return is_acceptable


# Context managers: guarantee setup/cleanup around resources.
@contextmanager
def pipeline_run(name):
    print(f"Starting {name}")
    try:
        yield
    finally:
        print(f"Finished {name}")


# *args/**kwargs and unpacking: build flexible, composable APIs.
def report(title, *values, precision=2, **metadata):
    rounded = [round(value, precision) for value in values]
    return {"title": title, "values": rounded, **metadata}


# Lambda and sorting/key functions: express small selection rules.
runs = [
    {"id": "run-c", "loss": 0.31},
    {"id": "run-a", "loss": 0.18},
    {"id": "run-b", "loss": 0.24},
]
best_first = sorted(runs, key=lambda run: (run["loss"], run["id"]))


# collections: specialized containers for counting, grouping, and queues.
labels = Counter(["cat", "dog", "cat", "bird"])
by_status = defaultdict(list)
by_status["success"].extend(["run-a", "run-b"])
recent_runs = deque(["run-a", "run-b"], maxlen=3)
recent_runs.append("run-c")


# itertools: combine or slice lazy iterables efficiently.
parameter_pairs = list(combinations([0.01, 0.1, 1.0], 2))
first_three = list(islice((number * number for number in range(100)), 3))
all_ids = list(chain(["train-1"], ["test-1"], ["deploy-1"]))


# functools: cache, specialize, and reduce functions.
@lru_cache(maxsize=128)
def expensive_feature(feature_id):
    return feature_id * feature_id


round_to_three = partial(round, ndigits=3)
total_samples = reduce(lambda total, value: total + value, [120, 80, 50], 0)


@timed
def demo():
    is_fast_enough = threshold_checker(50)
    configuration = {"environment": "staging", "owner": "ml-team"}

    with pipeline_run("advanced-python-demo"):
        print("Comprehensions:", squared_ids, good_metrics, unique_stages)
        print("Generators/yield:", list(batches(list(range(7)), 3)))
        print("Iterator:", list(Countdown(3)))
        print("Closure:", is_fast_enough(metrics["latency_ms"]))
        print("args/kwargs + unpacking:", report("scores", 0.9345, 0.8123, **configuration))
        print("Lambda/key sorting:", best_first)
        print("collections:", labels, dict(by_status), list(recent_runs))
        print("itertools:", parameter_pairs, first_three, all_ids)
        print("functools:", expensive_feature(12), round_to_three(3.14159), total_samples)


if __name__ == "__main__":
    demo()
