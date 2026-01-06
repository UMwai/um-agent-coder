"""
Data validation utilities.

This module provides small, dependency-free validators for common data types
(email, URL, date/time) plus a lightweight JSON-Schema-like validator and a
composable validator builder.

Examples:
    >>> is_valid_email("dev@example.com")
    True
    >>> is_valid_url("https://example.com/path?q=1")
    True
    >>> validate_json_string('{"age": 3}', {"type": "object", "properties": {"age": {"type": "integer"}}})
    {'age': 3}

    >>> v = non_empty_string().then(matches_regex(r"^\\d{4}$", "4 digits")).map(int)
    >>> v("2025")
    2025
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone, tzinfo
from ipaddress import ip_address
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from urllib.parse import urlsplit

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class ValidationError(ValueError):
    """Raised when a value fails validation.

    Attributes:
        message: Human readable error message.
        path: A JSONPath-like location for structured validators (default: "$").
        value: The offending value (kept as-is).

    Examples:
        >>> try:
        ...     validate_email("not-an-email")
        ... except ValidationError as e:
        ...     "email" in str(e).lower()
        True
    """

    message: str
    path: str = "$"
    value: Any = None

    def __str__(self) -> str:  # pragma: no cover (trivial)
        if self.path and self.path != "$":
            return f"{self.message} at {self.path}"
        return self.message


class SchemaValidationError(ValidationError):
    """Raised when JSON schema validation fails."""


class DateTimeValidationError(ValidationError):
    """Raised when date/time parsing or validation fails."""


_EMAIL_RE = re.compile(
    # Pragmatic RFC 5322-ish regex: allows most common local-part chars and enforces dot-separated domain.
    r"^(?=.{1,254}$)(?P<local>[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64})@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63})$"
)


def is_valid_email(email: Any) -> bool:
    """Return True if `email` looks like a valid email address.

    Notes:
        This is a format check (regex + a few pragmatic constraints), not a deliverability check.

    Examples:
        >>> is_valid_email("dev@example.com")
        True
        >>> is_valid_email("dev@localhost")
        False
    """

    if not isinstance(email, str):
        return False
    candidate = email.strip()
    match = _EMAIL_RE.fullmatch(candidate)
    if not match:
        return False
    local = match.group("local")
    if ".." in local or local.startswith(".") or local.endswith("."):
        return False
    return True


def validate_email(email: Any) -> str:
    """Validate email format and return the normalized (trimmed) email.

    Raises:
        ValidationError: if the email is not valid.

    Examples:
        >>> validate_email("  dev@example.com ")
        'dev@example.com'
    """

    if not isinstance(email, str):
        raise ValidationError("Email must be a string", path="$", value=email)
    candidate = email.strip()
    if not is_valid_email(candidate):
        raise ValidationError("Invalid email format", path="$", value=email)
    return candidate


def _is_valid_hostname(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        ip_address(hostname)
        return True
    except ValueError:
        pass
    if len(hostname) > 253:
        return False
    labels = hostname.split(".")
    if len(labels) < 2:
        return False
    label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
    return all(label_re.fullmatch(label) is not None for label in labels)


def is_valid_url(url: Any, allowed_schemes: Collection[str] = ("http", "https")) -> bool:
    """Return True if `url` looks like a valid URL.

    Examples:
        >>> is_valid_url("https://example.com")
        True
        >>> is_valid_url("example.com")
        False
    """

    if not isinstance(url, str):
        return False
    candidate = url.strip()
    if not candidate or any(ch.isspace() for ch in candidate):
        return False
    try:
        parsed = urlsplit(candidate)
    except Exception:
        return False
    if parsed.scheme.lower() not in {s.lower() for s in allowed_schemes}:
        return False
    if not parsed.netloc:
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    if not _is_valid_hostname(hostname):
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    if port is not None and not (0 < port < 65536):
        return False
    return True


def validate_url(url: Any, allowed_schemes: Collection[str] = ("http", "https")) -> str:
    """Validate URL format and return the normalized (trimmed) URL.

    Raises:
        ValidationError: if the URL is not valid.

    Examples:
        >>> validate_url(" https://example.com/a ")
        'https://example.com/a'
    """

    if not isinstance(url, str):
        raise ValidationError("URL must be a string", path="$", value=url)
    candidate = url.strip()
    if not is_valid_url(candidate, allowed_schemes=allowed_schemes):
        raise ValidationError("Invalid URL format", path="$", value=url)
    return candidate


def validate_json_string(json_text: str, schema: Mapping[str, Any]) -> Any:
    """Parse `json_text` and validate the resulting object against `schema`.

    Examples:
        >>> validate_json_string('{"x": 1}', {"type": "object", "properties": {"x": {"type": "integer"}}})
        {'x': 1}
    """

    try:
        instance = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e.msg}", path="$", value=json_text) from e
    validate_json_schema(instance, schema)
    return instance


def is_valid_json_schema(instance: Any, schema: Mapping[str, Any]) -> bool:
    """Return True if `instance` validates against `schema`."""

    try:
        validate_json_schema(instance, schema)
        return True
    except SchemaValidationError:
        return False


def _schema_type_matches(instance: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(instance, dict)
    if schema_type == "array":
        return isinstance(instance, list)
    if schema_type == "string":
        return isinstance(instance, str)
    if schema_type == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if schema_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if schema_type == "boolean":
        return isinstance(instance, bool)
    if schema_type == "null":
        return instance is None
    return False


def _type_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def validate_json_schema(instance: Any, schema: Mapping[str, Any], *, path: str = "$") -> None:
    """Validate `instance` against a lightweight JSON-Schema-like `schema`.

    Supported keywords (subset): `type`, `properties`, `required`, `items`, `enum`, `pattern`,
    `minLength`, `maxLength`, `minimum`, `maximum`, `minItems`, `maxItems`,
    `additionalProperties`, `anyOf`, `allOf`, `oneOf`, and `format` (email/uri/date/date-time).

    Examples:
        >>> schema = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}
        >>> validate_json_schema({"id": "abc"}, schema)
        >>> validate_json_schema({}, schema)
        Traceback (most recent call last):
        ...
        SchemaValidationError: Missing required property 'id'
    """

    if not isinstance(schema, Mapping):
        raise SchemaValidationError("Schema must be a mapping", path=path, value=schema)

    if "anyOf" in schema:
        errors: List[str] = []
        for option in schema.get("anyOf", []):
            try:
                validate_json_schema(instance, option, path=path)
                break
            except SchemaValidationError as e:
                errors.append(str(e))
        else:
            raise SchemaValidationError("Value does not match anyOf", path=path, value=instance)

    if "allOf" in schema:
        for option in schema.get("allOf", []):
            validate_json_schema(instance, option, path=path)

    if "oneOf" in schema:
        matches = 0
        last_error: Optional[SchemaValidationError] = None
        for option in schema.get("oneOf", []):
            try:
                validate_json_schema(instance, option, path=path)
                matches += 1
            except SchemaValidationError as e:
                last_error = e
        if matches != 1:
            raise SchemaValidationError(
                "Value does not match exactly oneOf", path=path, value=instance
            ) from last_error

    if "enum" in schema:
        if instance not in schema["enum"]:
            raise SchemaValidationError("Value not in enum", path=path, value=instance)

    if "type" in schema:
        schema_type = schema["type"]
        if isinstance(schema_type, str):
            ok = _schema_type_matches(instance, schema_type)
            expected = schema_type
        elif isinstance(schema_type, Sequence):
            expected_types = [t for t in schema_type if isinstance(t, str)]
            ok = any(_schema_type_matches(instance, t) for t in expected_types)
            expected = " | ".join(expected_types) if expected_types else "unknown"
        else:
            raise SchemaValidationError("Invalid schema 'type'", path=path, value=schema_type)
        if not ok:
            raise SchemaValidationError(
                f"Expected type {expected}, got {_type_label(instance)}",
                path=path,
                value=instance,
            )

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            raise SchemaValidationError("String shorter than minLength", path=path, value=instance)
        if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
            raise SchemaValidationError("String longer than maxLength", path=path, value=instance)
        if "pattern" in schema:
            pattern = str(schema["pattern"])
            if re.search(pattern, instance) is None:
                raise SchemaValidationError("String does not match pattern", path=path, value=instance)
        if "format" in schema:
            fmt = str(schema["format"])
            if fmt == "email" and not is_valid_email(instance):
                raise SchemaValidationError("String is not a valid email", path=path, value=instance)
            if fmt == "uri" and not is_valid_url(instance, allowed_schemes=("http", "https")):
                raise SchemaValidationError("String is not a valid URI", path=path, value=instance)
            if fmt == "date":
                try:
                    parse_date(instance)
                except DateTimeValidationError as e:
                    raise SchemaValidationError("String is not a valid date", path=path, value=instance) from e
            if fmt in {"date-time", "datetime"}:
                try:
                    parse_datetime(instance)
                except DateTimeValidationError as e:
                    raise SchemaValidationError(
                        "String is not a valid date-time", path=path, value=instance
                    ) from e

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < float(schema["minimum"]):
            raise SchemaValidationError("Number less than minimum", path=path, value=instance)
        if "maximum" in schema and instance > float(schema["maximum"]):
            raise SchemaValidationError("Number greater than maximum", path=path, value=instance)

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            raise SchemaValidationError("Array shorter than minItems", path=path, value=instance)
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            raise SchemaValidationError("Array longer than maxItems", path=path, value=instance)
        if "items" in schema:
            item_schema = schema["items"]
            for idx, item in enumerate(instance):
                validate_json_schema(item, item_schema, path=f"{path}[{idx}]")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        if required:
            if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
                raise SchemaValidationError("Schema 'required' must be a list", path=path, value=required)
            for key in required:
                if not isinstance(key, str):
                    raise SchemaValidationError(
                        "Schema 'required' must contain strings", path=path, value=required
                    )
                if key not in instance:
                    raise SchemaValidationError(f"Missing required property '{key}'", path=path, value=instance)

        properties: Mapping[str, Any] = schema.get("properties", {}) or {}
        if properties and not isinstance(properties, Mapping):
            raise SchemaValidationError(
                "Schema 'properties' must be a mapping", path=path, value=properties
            )

        for key, prop_schema in properties.items():
            if key in instance:
                validate_json_schema(instance[key], prop_schema, path=f"{path}.{key}")

        additional = schema.get("additionalProperties", True)
        if additional is False:
            allowed = set(properties.keys())
            for key in instance.keys():
                if key not in allowed:
                    raise SchemaValidationError(
                        f"Additional property '{key}' not allowed", path=path, value=instance
                    )
        elif isinstance(additional, Mapping):
            for key, value in instance.items():
                if key not in properties:
                    validate_json_schema(value, additional, path=f"{path}.{key}")


_DEFAULT_DATE_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
    "%d-%m-%Y",
    "%m/%d/%Y",
)

_DEFAULT_DATETIME_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
)


def parse_date(value: Union[str, date, datetime], *, formats: Optional[Sequence[str]] = None) -> date:
    """Parse a date from common string formats.

    Supported formats include ISO dates and a few pragmatic variants. If `value` is already a
    `date`/`datetime`, it is converted to `date`.

    Examples:
        >>> parse_date("2025-01-03").isoformat()
        '2025-01-03'
        >>> parse_date(datetime(2025, 1, 3, 12, 0)).isoformat()
        '2025-01-03'
    """

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise DateTimeValidationError("Date must be a string or date-like", path="$", value=value)

    candidate = value.strip()
    if not candidate:
        raise DateTimeValidationError("Date string is empty", path="$", value=value)

    # First try ISO 8601 date.
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        pass

    for fmt in formats or _DEFAULT_DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    raise DateTimeValidationError("Unrecognized date format", path="$", value=value)


def _normalize_iso_datetime(text: str) -> str:
    normalized = text.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return normalized


def parse_datetime(
    value: Union[str, date, datetime],
    *,
    formats: Optional[Sequence[str]] = None,
    assume_tz: Optional[tzinfo] = timezone.utc,
) -> datetime:
    """Parse a datetime from common string formats (including ISO 8601).

    For ISO strings, timezone offsets are supported. If no timezone is present and `assume_tz`
    is provided (default UTC), the resulting datetime is made timezone-aware.

    Examples:
        >>> parse_datetime("2025-01-03T12:30:00Z").tzinfo is not None
        True
        >>> parse_datetime("2025-01-03 12:30").hour
        12
    """

    if isinstance(value, datetime):
        return value if value.tzinfo is not None or assume_tz is None else value.replace(tzinfo=assume_tz)
    if isinstance(value, date):
        dt = datetime.combine(value, time.min)
        return dt if assume_tz is None else dt.replace(tzinfo=assume_tz)
    if not isinstance(value, str):
        raise DateTimeValidationError("Datetime must be a string or date-like", path="$", value=value)

    candidate = value.strip()
    if not candidate:
        raise DateTimeValidationError("Datetime string is empty", path="$", value=value)

    # ISO 8601 datetime.
    try:
        dt = datetime.fromisoformat(_normalize_iso_datetime(candidate))
        if dt.tzinfo is None and assume_tz is not None:
            dt = dt.replace(tzinfo=assume_tz)
        return dt
    except ValueError:
        pass

    for fmt in formats or _DEFAULT_DATETIME_FORMATS:
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt if assume_tz is None else dt.replace(tzinfo=assume_tz)
        except ValueError:
            continue
    raise DateTimeValidationError("Unrecognized datetime format", path="$", value=value)


def is_valid_date(value: Any, *, formats: Optional[Sequence[str]] = None) -> bool:
    """Return True if `value` can be parsed as a date."""

    try:
        parse_date(value, formats=formats)
        return True
    except DateTimeValidationError:
        return False


def is_valid_datetime(value: Any, *, formats: Optional[Sequence[str]] = None) -> bool:
    """Return True if `value` can be parsed as a datetime."""

    try:
        parse_datetime(value, formats=formats)
        return True
    except DateTimeValidationError:
        return False


class Validator(Generic[T]):
    """Composable validator.

    A `Validator` is a callable that either returns a validated/transformed value or raises
    `ValidationError`.

    Examples:
        >>> v = non_empty_string().then(matches_regex(r"^\\d+$", "digits")).map(int)
        >>> v("123")
        123
    """

    def __init__(self, func: Callable[[Any], T], *, name: str = "value"):
        self._func = func
        self._name = name

    def with_name(self, name: str) -> "Validator[T]":
        """Return a copy that uses `name` in error messages."""

        return Validator(self._func, name=name)

    def validate(self, value: Any) -> T:
        """Validate `value` or raise `ValidationError`."""

        try:
            return self._func(value)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(str(e), path="$", value=value) from e

    def __call__(self, value: Any) -> T:
        return self.validate(value)

    def then(self, next_validator: "Validator[U]") -> "Validator[U]":
        """Compose this validator with another."""

        def _composed(value: Any) -> U:
            return next_validator(self.validate(value))

        return Validator(_composed, name=self._name)

    def map(self, mapper: Callable[[T], U]) -> "Validator[U]":
        """Transform the validated value."""

        def _mapped(value: Any) -> U:
            return mapper(self.validate(value))

        return Validator(_mapped, name=self._name)

    def optional(self) -> "Validator[Optional[T]]":
        """Accept `None` without further validation."""

        def _optional(value: Any) -> Optional[T]:
            if value is None:
                return None
            return self.validate(value)

        return Validator(_optional, name=self._name)

    def or_else(self, fallback: "Validator[T]") -> "Validator[T]":
        """Try this validator, otherwise try `fallback`."""

        def _either(value: Any) -> T:
            try:
                return self.validate(value)
            except ValidationError:
                return fallback.validate(value)

        return Validator(_either, name=self._name)


def compose(*validators: Validator[Any]) -> Validator[Any]:
    """Compose validators left-to-right.

    Examples:
        >>> v = compose(non_empty_string(), matches_regex(r"^a+$", "only a"))
        >>> v("aaa")
        'aaa'
    """

    if not validators:
        return Validator(lambda x: x)
    out: Validator[Any] = validators[0]
    for v in validators[1:]:
        out = out.then(v)
    return out


def predicate(check: Callable[[Any], bool], message: str) -> Validator[Any]:
    """Build a validator from a boolean predicate."""

    def _pred(value: Any) -> Any:
        if not check(value):
            raise ValidationError(message, path="$", value=value)
        return value

    return Validator(_pred)


def instance_of(expected_type: Union[Type[Any], Tuple[Type[Any], ...]]) -> Validator[Any]:
    """Validate that a value is an instance of `expected_type`."""

    def _inst(value: Any) -> Any:
        if not isinstance(value, expected_type):
            raise ValidationError(f"Expected {expected_type}, got {type(value)}", path="$", value=value)
        return value

    return Validator(_inst)


def non_empty_string(*, strip: bool = True) -> Validator[str]:
    """Validate that a value is a non-empty string.

    Examples:
        >>> non_empty_string()(" hi ")
        'hi'
    """

    def _s(value: Any) -> str:
        if not isinstance(value, str):
            raise ValidationError("Expected a string", path="$", value=value)
        out = value.strip() if strip else value
        if not out:
            raise ValidationError("String must not be empty", path="$", value=value)
        return out

    return Validator(_s)


def matches_regex(pattern: Union[str, re.Pattern[str]], message: str = "Invalid format") -> Validator[str]:
    """Validate a string by regex."""

    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    def _rx(value: Any) -> str:
        s = non_empty_string()(value)
        if compiled.search(s) is None:
            raise ValidationError(message, path="$", value=value)
        return s

    return Validator(_rx)


def one_of(options: Collection[T], *, message: str = "Value not allowed") -> Validator[T]:
    """Validate that the value is in `options`."""

    def _one(value: Any) -> T:
        if value not in options:
            raise ValidationError(message, path="$", value=value)
        return value  # type: ignore[return-value]

    return Validator(_one)


def min_length(min_len: int) -> Validator[str]:
    """Validate that a string has at least `min_len` characters."""

    def _min(value: Any) -> str:
        s = non_empty_string(strip=False)(value)
        if len(s) < min_len:
            raise ValidationError(f"String shorter than {min_len}", path="$", value=value)
        return s

    return Validator(_min)


def max_length(max_len: int) -> Validator[str]:
    """Validate that a string has at most `max_len` characters."""

    def _max(value: Any) -> str:
        s = non_empty_string(strip=False)(value)
        if len(s) > max_len:
            raise ValidationError(f"String longer than {max_len}", path="$", value=value)
        return s

    return Validator(_max)


def in_range(
    minimum: Optional[float] = None, maximum: Optional[float] = None
) -> Validator[Union[int, float]]:
    """Validate that a numeric value falls within bounds."""

    def _rng(value: Any) -> Union[int, float]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationError("Expected a number", path="$", value=value)
        if minimum is not None and value < minimum:
            raise ValidationError(f"Number less than {minimum}", path="$", value=value)
        if maximum is not None and value > maximum:
            raise ValidationError(f"Number greater than {maximum}", path="$", value=value)
        return value

    return Validator(_rng)
