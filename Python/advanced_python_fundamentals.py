"""Advanced Python fundamentals through practical MLOps-style examples.

Use these tools to write pipelines that are readable, memory-efficient,
reusable, and easy to instrument. Run this file to see each example.

Requires Python 3.10+. Core demos use the standard library. Optional API demo:
    python -m pip install 'pydantic>=2,<3' 'fastapi>=0.115,<1' uvicorn
    python advanced_python_fundamentals.py --pydantic
    python -m uvicorn advanced_python_fundamentals:create_app --factory
Run the uvicorn command from this file's directory, then visit /docs.
Optional pytest lesson (fixtures, parameterization, mocks, unit/integration/API tests):
    python -m pip install pytest fastapi "pydantic>=2,<3" httpx
    python advanced_python_fundamentals.py --pytest
Optional logging/configuration lesson:
    python -m pip install 'pydantic-settings>=2,<3' 'PyYAML>=6,<7'
    python advanced_python_fundamentals.py --operations
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, partial, reduce, wraps
from itertools import chain, combinations, islice
from math import isfinite
from time import perf_counter
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Generic, Literal, Optional, Protocol, TypeVar, TypedDict, Union


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


# Typing and clean application design
# typing describes contracts for editors/type checkers; Python does not enforce
# annotations by itself. Prefer precise types over Any (which disables checks).
# object means an unknown value that must be narrowed before using it.
# References: https://docs.python.org/3/library/typing.html
# https://docs.pydantic.dev/latest/concepts/fields/
# https://fastapi.tiangolo.com/tutorial/dependencies/


# Optional[T] means T | None, NOT 'the argument can be omitted'. A default makes
# an argument omittable. Union[A, B] means A | B; narrow it with isinstance.
def normalize_run_id(value: Union[str, int]) -> str:
    """Pure function: same input/output, no I/O, no mutation of caller data."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError("run_id must be a string or integer, excluding bool")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("run_id must not be empty")
    return normalized


def find_run_index(run_ids: Sequence[str], target: str) -> Optional[int]:
    """Return None for absence; zero is a valid index, so don't test truthiness."""
    for index, run_id in enumerate(run_ids):
        if run_id == target:
            return index
    return None


# Literal narrows values, not just types. Runtime callers still need validation.
RunStage = Literal["train", "evaluate", "serve"]


class RunSummary(TypedDict):
    """A plain dictionary's static shape; this creates no runtime validator."""
    run_id: str
    stage: RunStage
    score: Optional[float]  # Required key whose VALUE may be None.


def summarize_run(run_id: Union[str, int], stage: RunStage) -> RunSummary:
    if stage not in ("train", "evaluate", "serve"):
        raise ValueError("Unsupported run stage")
    return {"run_id": normalize_run_id(run_id), "stage": stage, "score": None}


# Generics preserve relationships between input/output types. T is a placeholder,
# unlike Any. Page[Prediction] and Page[RunSummary] share code but retain item types.
# TypeVar/Generic syntax works on Python 3.10; class Page[T] requires Python 3.12+.
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    items: tuple[T, ...]

    def first(self) -> Optional[T]:
        return self.items[0] if self.items else None


# Protocol again, at the application boundary: HTTP code needs only predict().
# An implementation can satisfy this interface without inheriting this class.
class Predictor(Protocol):
    def predict(self, features: tuple[float, ...]) -> Prediction:
        ...


def load_model_config(environ: Mapping[str, str]) -> ModelConfig:
    """Read an explicit environment snapshot; validate before serving requests."""
    # Environment variables are strings: parsing is deliberate, unlike arbitrary
    # coercion of request JSON. Passing a mapping makes tests independent of os.
    raw_threshold = environ.get("ML_THRESHOLD", "0.7")
    try:
        threshold = float(raw_threshold)
    except ValueError as exc:
        raise ValueError("ML_THRESHOLD must be a number between 0 and 1") from exc
    return ModelConfig(environ.get("ML_MODEL_NAME", "mean-v1"), threshold)


# Configuration management: defaults < explicit environment values here. Load
# once at application creation, fail early, and inject validated configuration.
# Do not scatter os.getenv calls through business logic or commit secrets.
# Larger apps can use pydantic_settings.BaseSettings (a separate package in v2)
# for typed environment/.env/secrets sources; document precedence and never log
# the full settings object when it contains credentials.


def build_api_models():
    """Return Pydantic v2 schemas; defer optional imports for the core demo."""
    from pydantic import BaseModel, ConfigDict, Field

    # Annotated adds metadata; Pydantic interprets Field constraints at runtime.
    # DRY: define the probability contract once for both request and response.
    Probability = Annotated[
        float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    ]

    class PredictRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        features: list[Probability] = Field(min_length=1, max_length=1024)
        # Nullable AND omittable because it has a default. In Pydantic v2,
        # Optional[str] without '= None' would be required but accept null.
        request_id: Optional[str] = Field(default=None, max_length=100, strict=True)

    class PredictResponse(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        model_name: str
        score: Probability
        label: bool
        request_id: Optional[str] = None

    return PredictRequest, PredictResponse


def demo_pydantic() -> None:
    """Validate untrusted data, then serialize the validated model."""
    from pydantic import ValidationError

    request_model, _ = build_api_models()
    request = request_model.model_validate({"features": [0.8, 0.9]})
    print("Pydantic validated data:", request.model_dump())
    print("Pydantic JSON:", request.model_dump_json())
    try:
        request_model.model_validate({"features": ["0.8"], "typo": True})
    except ValidationError as exc:
        # Show just locations/types; raw validation errors may contain user data.
        print("Rejected fields:", [(error["loc"], error["type"]) for error in exc.errors()])


def create_app(service_factory: Optional[Callable[[], Predictor]] = None):
    """Build the optional FastAPI app; callers can inject a test service factory."""
    from fastapi import Depends, FastAPI, HTTPException

    request_model, response_model = build_api_models()
    app = FastAPI(title="Typed prediction example")

    if service_factory is None:
        config = load_model_config(dict(os.environ))
        model = MeanProbabilityModel(config)

        def default_service_factory() -> Predictor:
            # Per-request demo sink avoids unbounded shared in-memory history.
            # It is intentionally ephemeral; inject a durable adapter in real use.
            return PredictionService(config, model, InMemoryPredictionSink())

        service_factory = default_service_factory

    # Separate concerns: schemas validate HTTP data; the service runs business
    # logic; the route maps inputs/outputs. Domain code imports no FastAPI classes.
    # Use def for this synchronous service. Blocking work in async def blocks the
    # event loop; heavy inference usually needs a dedicated execution strategy.
    @app.post("/predict", response_model=response_model)
    def predict_route(
        payload: request_model,
        service: Annotated[Predictor, Depends(service_factory)],
    ):
        try:
            result = service.predict(tuple(payload.features))
        except OSError as exc:
            # Translate this known infrastructure failure, not every exception.
            # Invalid server/model output should remain a server error, not 422.
            raise HTTPException(status_code=503, detail="Prediction storage unavailable") from exc
        return response_model(
            model_name=result.model_name,
            score=result.score,
            label=result.label,
            request_id=payload.request_id,
        )

    # POST /predict {"features": [0.8, 0.9]} -> 200 with score ~0.85.
    # Empty/out-of-range/string features or unknown keys -> 422 before service use.
    # FastAPI derives OpenAPI docs and validates responses using these schemas.
    return app


def demo_typing() -> None:
    summary = summarize_run(42, "evaluate")
    page: Page[RunSummary] = Page((summary,))
    index = find_run_index(["42", "43"], "42")
    print("\nTyping / Union / Literal / TypedDict:", summary)
    print("Optional (zero is present):", index if index is not None else "missing")
    print("Generics preserve the item type:", page.first())
    print("Validated configuration:", load_model_config({"ML_THRESHOLD": "0.8"}))


# Clean functions: one clear job, explicit parameters/return values, descriptive
# names, predictable errors, no hidden I/O or mutation (see normalize_run_id).
# Separation of concerns: parsing, prediction, persistence, and HTTP are separate.
# DRY: share stable rules (finite_number, Probability), not merely similar-looking
# code. Validation at both an HTTP boundary and a reusable domain boundary is
# intentional: non-HTTP callers also need domain invariants.
# SOLID basics applied to this example:
# S - Single responsibility: models score, sinks record, routes handle HTTP.
# O - Open/closed: introduce a new ScoringModel or sink without editing the service.
# L - Liskov substitution: every ScoringModel must preserve score's input/output
#     promises; do not return None or silently change the meaning of scores.
# I - Interface segregation: Predictor and PredictionSink expose only what each
#     consumer needs, instead of one large train/save/serve interface.
# D - Dependency inversion: the route depends on Predictor, and PredictionService
#     on ScoringModel/PredictionSink abstractions. Injection supplies concrete ones.
# Modularization when this learning file grows into an application:
#   domain.py        -> Prediction, ModelConfig, ScoringModel, PredictionSink
#   services.py      -> PredictionService (imports domain, never routes)
#   adapters.py      -> model/storage implementations (imports domain)
#   schemas.py       -> Pydantic request/response models, normally module-level
#   settings.py      -> environment parsing and validated settings
#   api.py           -> routes and dependency providers
#   main.py          -> composition root, lifespan resource setup/cleanup
#   tests/           -> pure unit tests and HTTP contract tests
# Keep these in one file here so the lesson runs directly. Schema factories defer
# optional imports; ordinary API projects should define schemas at module level
# for better static analysis. Use a type checker in CI alongside runtime tests.
# Pin tested dependency versions in your application's lockfile. This educational
# API has no authentication or durable storage and is not a deployment template.


# Logging, exceptions, and layered configuration
# References: https://docs.python.org/3/howto/logging.html
# https://docs.pydantic.dev/latest/concepts/pydantic_settings/
# https://pyyaml.org/wiki/PyYAMLDocumentation


class ApplicationError(Exception):
    """Base for expected application failures; do not catch BaseException."""


class ConfigurationError(ApplicationError):
    """Configuration cannot be read or violates the application's schema."""


# Custom exceptions let callers recover by meaning rather than parsing messages.
# `raise ... from exc` retains the cause for debugging. Catch specific failures at
# the boundary that can handle them; do not silently return defaults after errors.


class JsonEventFormatter(logging.Formatter):
    """One JSON object per record, with a small allowlist of contextual fields."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        # These fields must contain approved, nonsensitive values. An allowlist is
        # not automatic redaction: never put a token in event or request_id either.
        for key in ("request_id", "error_code"):
            value = getattr(record, key, None)
            if isinstance(value, str):
                event[key] = value
        if record.exc_info:
            # Exception messages/tracebacks may contain credentials or payloads.
            # Deliberately emit only the type; full traces belong in a protected
            # diagnostic channel with an explicit redaction/access policy.
            event["exception_type"] = record.exc_info[0].__name__
        return json.dumps(event, ensure_ascii=True, allow_nan=False)


def configure_event_logging(level: str = "INFO", stream=None) -> logging.Logger:
    """Configure only this application's logger, at startup, not per request."""
    levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in levels:
        raise ConfigurationError("Unsupported logging level")
    logger = logging.getLogger("advanced_python.operations")
    logger.setLevel(level)
    logger.propagate = False  # Prevent a second copy through root handlers.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)
    return logger


# Levels (in increasing severity):
# DEBUG: detailed diagnostics, normally disabled in production.
# INFO: expected milestones, such as configuration_loaded.
# WARNING: recoverable degradation, such as a fallback being used.
# ERROR: an operation failed; the process may still serve other requests.
# CRITICAL: the application cannot continue, such as failed startup.
# A logger set to INFO suppresses DEBUG. Handlers can also filter levels.
# Libraries obtain logging.getLogger(__name__); the executable owns handlers.
# Use logger.info("processed %s records", count) for lazy interpolation. Structured
# events use stable names plus extra fields so collectors can search/aggregate.
# logger.exception(...) is ERROR with exc_info=True; use it inside an except block.


def read_config_file(path: Path) -> dict[str, object]:
    """Read a small, trusted operator-supplied JSON or YAML configuration file."""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ConfigurationError("Configuration must use .json, .yaml, or .yml")
    try:
        # Bound the read before decoding. Safe YAML loading prevents Python object
        # construction, but does not make arbitrary hostile input resource-safe.
        with path.open("rb") as source:
            raw = source.read(65_537)
        if len(raw) > 65_536:
            raise ConfigurationError("Configuration exceeds 64 KiB")
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError("Cannot read UTF-8 configuration") from exc
    if suffix == ".json":
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ConfigurationError("Invalid JSON configuration") from exc
    else:
        import yaml

        try:
            data = yaml.safe_load(text)  # Never yaml.load with an unsafe loader.
        except yaml.YAMLError as exc:
            raise ConfigurationError("Invalid YAML configuration") from exc
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ConfigurationError("Configuration must be a mapping with string keys")
    if "api_token" in data:
        raise ConfigurationError("Supply api_token through environment or secret files")
    return data


def build_settings_model(file_values: Mapping[str, object]):
    """Create optional Pydantic v2 settings with explicit source precedence."""
    from pydantic import Field, SecretStr, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class ServiceSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="STUDY_", extra="forbid", frozen=True,
            env_file_encoding="utf-8", hide_input_in_errors=True,
        )
        model_name: str = Field(default="mean-v1", min_length=1)
        threshold: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
        api_token: SecretStr  # Required: deliberately no embedded credential.

        @field_validator("model_name")
        @classmethod
        def nonblank_name(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("model_name must not be blank")
            return value

        @field_validator("api_token")
        @classmethod
        def nonblank_token(cls, value: SecretStr) -> SecretStr:
            if not value.get_secret_value().strip():
                raise ValueError("api_token must not be blank")
            return value

        @classmethod
        def settings_customise_sources(
            cls, settings_cls, init_settings, env_settings,
            dotenv_settings, file_secret_settings,
        ):
            # Highest -> lowest priority. File data is a separate source, NOT
            # constructor kwargs, so environment values can override the file.
            return (
                init_settings, env_settings, dotenv_settings,
                file_secret_settings, lambda: dict(file_values),
            )

    return ServiceSettings


def load_service_settings(
    config_path: Optional[Path] = None, *,
    dotenv_path: Optional[Path] = None, secrets_dir: Optional[Path] = None,
):
    """Resolve settings once at startup; never read .env implicitly on import."""
    from pydantic import ValidationError
    from pydantic_settings import SettingsError

    for path in (dotenv_path,):
        if path is not None and not path.is_file():
            raise ConfigurationError("Explicit .env file does not exist")
    if secrets_dir is not None and not secrets_dir.is_dir():
        raise ConfigurationError("Explicit secrets directory does not exist")
    values = read_config_file(config_path) if config_path is not None else {}
    settings_type = build_settings_model(values)
    try:
        return settings_type(_env_file=dotenv_path, _secrets_dir=secrets_dir)
    except (ValidationError, SettingsError, OSError, UnicodeError) as exc:
        # Do not expose raw error dictionaries: they can contain sensitive input.
        raise ConfigurationError("Service settings are missing or invalid") from exc


# Configuration formats, representing the same non-secret values:
# JSON: {"model_name": "mean-v1", "threshold": 0.8, "log_level": "INFO"}
# YAML (indentation matters; quote strings that resemble booleans/dates):
#   model_name: mean-v1
#   threshold: 0.8
#   log_level: INFO
# JSON is strict and portable; YAML supports comments and is convenient for humans.
# Both need schema validation AFTER parsing; neither should hold committed secrets.
#
# Environment variables are strings. Pydantic settings parses/validates them:
# PowerShell: $env:STUDY_THRESHOLD = "0.9"
# POSIX shell: export STUDY_THRESHOLD=0.9
# .env is a local plaintext convenience, NOT encryption or automatic shell config:
#   STUDY_THRESHOLD=0.85
#   STUDY_API_TOKEN=<supply-a-local-development-token>
# Pass dotenv_path=Path(".env") explicitly; existing environment values win.
# Ignore .env and secret directories in Git; commit only a placeholder .env.example.
# Mounted secret directory: a file named STUDY_API_TOKEN contains the token.
# Precedence here: constructor overrides > environment > .env > secret files >
# JSON/YAML > defaults. Unknown file/.env fields fail validation to catch typos.
# SecretStr masks ordinary representations; it does NOT encrypt memory/storage.
# Only call get_secret_value() at the external client boundary, never in logs.
# Prefer a managed secret store or restricted mounted secrets in deployments;
# rotate credentials, restrict access, and avoid dumping settings/environment.
# In FastAPI, load settings during app creation/lifespan and inject them or a
# configured client with Depends. Own client cleanup in lifespan. Cached settings
# do not automatically reload changed environment values or rotated credentials.


def demo_operations() -> None:
    """Exercise file formats, dotenv, secret files, validation, and JSON events."""
    logger = configure_event_logging()
    with TemporaryDirectory(prefix="python-settings-") as directory:
        root = Path(directory)
        config = root / "service.json"
        config.write_text(json.dumps({"threshold": 0.75}), encoding="utf-8")
        yaml_config = root / "service.yaml"
        yaml_config.write_text("threshold: 0.75\n", encoding="utf-8")
        dotenv = root / ".env"
        dotenv.write_text("STUDY_LOG_LEVEL=INFO\n", encoding="utf-8")
        secrets = root / "secrets"
        secrets.mkdir()
        # Only a fake demonstration value, never a real credential.
        (secrets / "STUDY_API_TOKEN").write_text("demo-only-not-a-credential", encoding="utf-8")
        settings = load_service_settings(config, dotenv_path=dotenv, secrets_dir=secrets)
        assert read_config_file(config) == read_config_file(yaml_config)
        logger = configure_event_logging(settings.log_level)
        logger.debug("configuration_diagnostics_ready")
        logger.info("configuration_loaded", extra={"request_id": "demo-1"})
        logger.warning("demo_storage_is_ephemeral")
        try:
            read_config_file(root / "missing.json")
        except ConfigurationError:
            logger.exception("configuration_read_failed", extra={"error_code": "CONFIG_READ"})
        # CRITICAL is reserved for a real inability to continue:
        # logger.critical("startup_failed", extra={"error_code": "CONFIG_INVALID"})
        print("Settings threshold:", settings.threshold)  # Explicit non-secret field.


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
        demo_typing()


# Pytest tutorial: executable examples for the application above.
# Install: python -m pip install pytest fastapi 'pydantic>=2,<3' httpx
# Run from this file's directory:
#   python advanced_python_fundamentals.py --pytest
#   python advanced_python_fundamentals.py --pytest -k unit -v
#   python advanced_python_fundamentals.py --pytest -k api --maxfail=1
# The runner writes this test module to a temporary directory. Keeping its source
# here makes the lesson self-contained without making pytest a core dependency.
# In a real project, move the string's contents into tests/test_predictions.py;
# place fixtures shared by several test modules in tests/conftest.py. Pytest finds
# test_*.py / *_test.py files and test_* functions; don't call fixtures yourself.
# References:
# https://docs.pytest.org/en/stable/how-to/fixtures.html
# https://docs.pytest.org/en/stable/how-to/parametrize.html
# https://docs.python.org/3/library/unittest.mock.html
# https://fastapi.tiangolo.com/tutorial/testing/
PYTEST_TUTORIAL = r'''
import os
from unittest.mock import Mock

import pytest
import advanced_python_fundamentals as lesson


# Fixtures: pytest injects values by argument name. Fixtures can depend on other
# fixtures. Function scope (the default) gives every test fresh mutable state.
# Other scopes: class, module, package, session. Broader scopes reuse resources,
# so avoid sharing mutable data. A broad fixture cannot request a narrower one.
# Prefer explicit fixture arguments over autouse=True when dependencies matter.
@pytest.fixture
def config():
    return lesson.ModelConfig("test-model", threshold=0.7)


@pytest.fixture
def sink():
    return lesson.InMemoryPredictionSink()


@pytest.fixture
def service(config, sink):
    return lesson.PredictionService(config, lesson.MeanProbabilityModel(config), sink)


# 1. Parameterized tests run once per case; ids give readable failure reports.
# Each expected value is a behavioral example, not a copy of implementation code.
@pytest.mark.parametrize("raw,expected", [
    pytest.param("  run-1  ", "run-1", id="trim-whitespace"),
    pytest.param(0, "0", id="zero-is-valid"),
    pytest.param(42, "42", id="integer-id"),
])
def test_unit_normalize_id(raw, expected):
    assert lesson.normalize_run_id(raw) == expected


# A parameterized fixture reruns EVERY dependent test for each fixture value.
@pytest.fixture(params=[0.0, 1.0], ids=["minimum-threshold", "maximum-threshold"])
def boundary_config(request):
    return lesson.ModelConfig("boundary-model", request.param)


def test_unit_threshold_endpoints(boundary_config):
    assert boundary_config.threshold in (0.0, 1.0)


# 2. Mocking replaces a collaborator, not the behavior being tested. spec_set
# rejects unknown attributes; side_effect simulates failures. Mock is from the
# standard library, so pytest-mock is not needed. For patch(), patch the name
# WHERE THE CODE UNDER TEST LOOKS IT UP, not necessarily where it was defined.
def test_unit_service_with_mock_model(config, sink):
    model = Mock(spec_set=lesson.ScoringModel)
    model.score.return_value = 0.9
    service = lesson.PredictionService(config, model, sink)
    result = service.predict((0.2, 0.4))
    assert result == lesson.Prediction("test-model", 0.9, True)
    model.score.assert_called_once_with((0.2, 0.4))
    assert sink.records == (result,)


def test_unit_storage_failure_propagates(config):
    sink = Mock(spec_set=lesson.InMemoryPredictionSink)
    sink.write.side_effect = OSError("demo disk unavailable")
    service = lesson.PredictionService(config, lesson.MeanProbabilityModel(config), sink)
    with pytest.raises(OSError, match="demo disk unavailable"):
        service.predict((0.8,))
    sink.write.assert_called_once()


# Built-in monkeypatch restores environment changes after the test, even on
# failure. This tests explicit environment parsing without editing real .env files.
def test_unit_environment_configuration(monkeypatch):
    monkeypatch.setenv("ML_MODEL_NAME", "isolated-model")
    monkeypatch.setenv("ML_THRESHOLD", "0.8")
    config = lesson.load_model_config(dict(os.environ))
    assert config == lesson.ModelConfig("isolated-model", 0.8)


# 3. Unit tests isolate a small contract. Arrange -> Act -> Assert makes intent
# visible. Use approx for floating-point calculations, not arbitrary rounding.
def test_unit_mean_score(config):
    model = lesson.MeanProbabilityModel(config)  # Arrange
    score = model.score((0.6, 0.8))              # Act
    assert score == pytest.approx(0.7)          # Assert


def test_unit_absent_index_is_not_zero():
    assert lesson.find_run_index(["run-a"], "run-a") == 0
    assert lesson.find_run_index(["run-a"], "missing") is None


# 4. Integration tests wire real components together. No model/sink mocks here.
# Unit vs integration describes the boundary, not a pytest-specific test type.
def test_integration_prediction_is_recorded(service, sink):
    result = service.predict((0.8, 0.9))
    assert result.score == pytest.approx(0.85)
    assert result.label is True
    assert sink.records == (result,)


# tmp_path is a per-test pathlib.Path. This exercises real file I/O + JSON
# parsing + configuration validation, entirely in a temporary directory.
def test_integration_config_file(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"model_name":"file-model","threshold":0.8}', encoding="utf-8")
    config = lesson.ModelConfig.from_mapping(lesson.read_config_file(path))
    assert config == lesson.ModelConfig("file-model", 0.8)


# 5. API tests verify HTTP status + response + side effects. TestClient runs the
# ASGI app in process, without a server or external network. This is not a deployed
# end-to-end test of DNS, TLS, proxies, authentication, or a production database.
@pytest.fixture
def client(service):
    # Skip only API tests if optional dependencies are missing; unit tests run.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    app = lesson.create_app(service_factory=lambda: service)
    # yield separates setup from cleanup. Context exit closes the client and runs
    # lifespan shutdown even if the test fails. Protect acquired resources with
    # context managers/finally: code after yield won't run if setup never yields.
    with TestClient(app) as test_client:
        yield test_client


def test_api_prediction(client, sink):
    response = client.post("/predict", json={"features": [0.8, 0.9], "request_id": "req-1"})
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "test-model", "score": pytest.approx(0.85),
        "label": True, "request_id": "req-1",
    }
    assert len(sink.records) == 1


@pytest.mark.parametrize("payload", [
    {}, {"features": []}, {"features": ["0.8"]},
    {"features": [True]}, {"features": [1.1]},
    {"features": [0.8], "unknown": 1}, {"features": [0.5] * 1025},
], ids=["missing", "empty", "string", "boolean", "out-of-range", "extra-field", "too-many"])
def test_api_invalid_input(client, sink, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]
    assert sink.records == ()  # Validation must not record a prediction.


def test_api_storage_failure(client, sink, monkeypatch):
    failing_write = Mock(side_effect=OSError("private infrastructure detail"))
    monkeypatch.setattr(sink, "write", failing_write)
    response = client.post("/predict", json={"features": [0.8]})
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction storage unavailable"}
    failing_write.assert_called_once()


# 6. Edge cases target contract boundaries: empty input, inclusive thresholds,
# invalid types (bool is an int subclass!), NaN/infinity, and missing files.
@pytest.mark.parametrize("value,error", [
    (True, TypeError), ("0.5", TypeError), (None, TypeError),
    (float("nan"), ValueError), (float("inf"), ValueError),
    (-0.01, ValueError), (1.01, ValueError),
])
def test_edge_invalid_threshold(value, error):
    with pytest.raises(error):
        lesson.ModelConfig("bad-config", value)


def test_edge_empty_features(service, sink):
    with pytest.raises(ValueError, match="At least one feature"):
        service.predict(())
    assert sink.records == ()


def test_edge_exact_threshold_is_positive(service):
    assert service.predict((0.7,)).label is True


def test_edge_missing_config_preserves_cause(tmp_path):
    with pytest.raises(lesson.ConfigurationError, match="Cannot read UTF-8") as caught:
        lesson.read_config_file(tmp_path / "missing.json")
    assert isinstance(caught.value.__cause__, FileNotFoundError)


# Avoid test-order dependencies, sleeps, real credentials, and external services.
# Assert observable behavior; don't mock every internal call. Use -k to select
# names here. For larger suites, register unit/integration/api markers in
# pytest.ini or pyproject.toml and select with -m. Skips are not passing coverage:
# install API dependencies in CI to exercise the HTTP tests too.
'''


def run_pytest_tutorial(arguments: list[str]) -> int:
    """Run the embedded test module in a fresh process; preserve pytest's exit code."""
    import subprocess

    with TemporaryDirectory(prefix="python-pytest-") as directory:
        tests = Path(directory) / "test_predictions.py"
        tests.write_text(PYTEST_TUTORIAL, encoding="utf-8")
        environment = dict(os.environ)
        module_directory = str(Path(__file__).resolve().parent)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (module_directory, environment.get("PYTHONPATH")))
        )
        return subprocess.run(
            [sys.executable, "-m", "pytest", str(tests), "-q",
             "-p", "no:cacheprovider", *arguments],
            env=environment, check=False,
        ).returncode



if __name__ == "__main__":
    if "--pytest" in sys.argv:
        raise SystemExit(run_pytest_tutorial(sys.argv[sys.argv.index("--pytest") + 1:]))
    demo()
    if "--pydantic" in sys.argv:
        demo_pydantic()
    if "--operations" in sys.argv:
        demo_operations()
