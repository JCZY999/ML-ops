"""Advanced Python fundamentals through practical MLOps-style examples.

Use these tools to write pipelines that are readable, memory-efficient,
reusable, and easy to instrument. Run this file to see each example.

Requires Python 3.10+; all examples use only the standard library.
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, partial, reduce, wraps
from itertools import chain, combinations, islice
from math import isfinite
from time import perf_counter
from typing import Protocol


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


# Object-oriented Python: a small, typed model-serving application.
# Read in order: value objects -> model -> interface -> service -> wiring.
# These examples demonstrate production design practices, not a complete serving
# platform. Deployment still needs authentication, monitoring, resource limits,
# concurrency policies, and a real trained model with a documented feature schema.


def finite_number(value: object, name: str) -> float:
    """Validate at an external boundary; annotations alone do not check values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float, excluding bool")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


# 1. Dataclasses: generate __init__, __repr__, and value-based __eq__ for data.
# frozen prevents normal field assignment; slots prevents accidental new fields.
# Frozen is shallow, so use immutable field values (here strings and floats).
# Dataclasses do not validate types: __post_init__ enforces our invariants.
@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_name: str
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        threshold = finite_number(self.threshold, "threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        # The supported initialization escape hatch for a frozen dataclass.
        object.__setattr__(self, "threshold", threshold)

    # 2. Class methods receive cls, and are useful for alternative constructors.
    # cls(...) preserves subclass construction; a hard-coded ModelConfig would not.
    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ModelConfig":
        """Parse configuration loaded from JSON or another mapping."""
        unknown = set(values) - {"model_name", "threshold"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        name = values.get("model_name")
        if not isinstance(name, str):
            raise TypeError("model_name must be a string")
        return cls(name, finite_number(values.get("threshold", 0.5), "threshold"))


@dataclass(frozen=True, slots=True)
class Prediction:
    model_name: str
    score: float
    label: bool


# 3. Abstract classes define a nominal contract: subclasses explicitly inherit it.
# ABC prevents instantiation until every abstract method is implemented. It does
# not check signature compatibility or return values; use a static type checker
# and behavioral tests too. A base class can also share concrete behavior.
class ScoringModel(ABC):
    @abstractmethod
    def score(self, features: tuple[float, ...]) -> float:
        """Return a finite score in [0, 1] for validated, nonempty features."""
        raise NotImplementedError


# 4. Classes bundle instance state and related behavior. self is this instance;
# __init__ initializes its state. _config signals internal use by convention,
# not enforced privacy. Keep mutable state on instances, not class attributes.
# 5. Inheritance expresses 'is a': MeanProbabilityModel is a ScoringModel.
# It honors the base contract, so callers can substitute it without special cases.
# Prefer shallow hierarchies; inheritance solely to reuse code adds coupling.
class MeanProbabilityModel(ScoringModel):
    """Teaching model: average probability features, not a trained predictor."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    # 6. Properties expose attribute syntax while controlling access. This cheap,
    # read-only property has no setter. Avoid hiding network I/O in properties.
    @property
    def config(self) -> ModelConfig:
        return self._config

    # 7. Static methods receive neither self nor cls. Use them for a stateless
    # operation naturally associated with a class; a module function is also fine.
    @staticmethod
    def validate_features(features: tuple[float, ...]) -> tuple[float, ...]:
        if not features:
            raise ValueError("At least one probability feature is required")
        validated = tuple(finite_number(value, "feature") for value in features)
        if any(not 0.0 <= value <= 1.0 for value in validated):
            raise ValueError("Probability features must be between 0 and 1")
        return validated

    def score(self, features: tuple[float, ...]) -> float:
        validated = self.validate_features(features)
        return sum(validated) / len(validated)


# 8. Protocols/interfaces define structural contracts: a compatible method is
# enough; explicit inheritance is unnecessary. Python has no interface keyword.
# Use an ABC for a controlled family/shared implementation; a Protocol for loosely
# coupled adapters, including third-party classes you cannot change.
# A type checker checks this contract. A Protocol is not runtime validation;
# even @runtime_checkable checks attribute presence, not signatures or semantics.
class PredictionSink(Protocol):
    def write(self, prediction: Prediction) -> None:
        """Record a prediction, or raise if recording fails."""
        ...


class InMemoryPredictionSink:
    """Per-instance storage for demos/tests; not durable or a concurrent queue."""

    def __init__(self) -> None:
        self._records: list[Prediction] = []

    def write(self, prediction: Prediction) -> None:
        self._records.append(prediction)

    @property
    def records(self) -> tuple[Prediction, ...]:
        # Return an immutable snapshot so callers cannot mutate our collection.
        return tuple(self._records)


# 9. Composition expresses 'has a': the service has a model and a sink. Neither
# needs to inherit from the service. Each collaborator has one responsibility.
# 10. Dependency injection means receiving collaborators instead of constructing
# them internally. Constructor injection makes requirements visible and lets tests
# supply fakes. No DI framework or global service locator is needed.
class PredictionService:
    def __init__(
        self, config: ModelConfig, model: ScoringModel, sink: PredictionSink
    ) -> None:
        self._config = config
        self._model = model
        self._sink = sink

    def predict(self, features: tuple[float, ...]) -> Prediction:
        # Validate before calling any injected implementation. The concrete model
        # additionally checks its own probability-feature schema.
        if not features:
            raise ValueError("At least one feature is required")
        validated = tuple(finite_number(value, "feature") for value in features)
        score = finite_number(self._model.score(validated), "model score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("Model score must be between 0 and 1")
        prediction = Prediction(
            model_name=self._config.model_name,
            score=score,
            label=score >= self._config.threshold,
        )
        # Explicit failure policy: a sink failure propagates to the caller.
        # Do not blindly retry this operation: a sink may write then raise, which
        # requires an idempotency strategy to avoid duplicate records in production.
        self._sink.write(prediction)
        return prediction


def demo_object_oriented_python() -> None:
    """Wire concrete dependencies at the application's composition root."""
    config = ModelConfig.from_mapping({"model_name": "mean-v1", "threshold": 0.7})
    model = MeanProbabilityModel(config)
    sink = InMemoryPredictionSink()
    service = PredictionService(model.config, model, sink)

    print("\nObject-oriented Python:")
    print("Class method + dataclass:", config)
    print("Read-only property:", model.config.model_name)
    print("Static method:", MeanProbabilityModel.validate_features((0.6, 0.8)))
    print("Inheritance + abstract class:", isinstance(model, ScoringModel))
    print("Composition + dependency injection:", service.predict((0.8, 0.9)))
    print("Protocol adapter + immutable snapshot:", sink.records)
    # score is approximately 0.85, label is True, and sink contains one record.


# Design guidance:
# - Use functions for stateless transformations; classes when state/behavior belong
#   together. A dataclass suits values; a regular class suits a service lifecycle.
# - Composition makes collaborators replaceable. Inheritance requires every
#   subclass to preserve its parent's behavioral promises (substitutability).
# - Validate untrusted data at boundaries, raise specific exceptions, and avoid
#   shared mutable defaults. For mutable dataclass fields use field(default_factory=...).
# - The composition root owns real clients/resources and their cleanup. Inject a
#   database or telemetry adapter implementing PredictionSink in a deployed app.
# - Type annotations help tools; tests verify behavior. Keep fake dependencies
#   deterministic and exercise invalid inputs and collaborator failures as well.


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
        demo_object_oriented_python()


if __name__ == "__main__":
    demo()

