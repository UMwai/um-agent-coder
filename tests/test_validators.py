import os
import sys
from datetime import date, datetime, timezone

import pytest

# Add src to path (repo doesn't always run tests with the package installed)
sys.path.append(os.path.join(os.getcwd(), "src"))

from um_agent_coder.utils.validators import (  # noqa: E402
    DateTimeValidationError,
    SchemaValidationError,
    ValidationError,
    compose,
    in_range,
    is_valid_date,
    is_valid_datetime,
    is_valid_email,
    is_valid_json_schema,
    is_valid_url,
    matches_regex,
    max_length,
    min_length,
    non_empty_string,
    one_of,
    parse_date,
    parse_datetime,
    validate_email,
    validate_json_schema,
    validate_json_string,
    validate_url,
)


def test_email_validator_valid_cases() -> None:
    assert is_valid_email("dev@example.com") is True
    assert is_valid_email("a.b+c_d@example.co.uk") is True
    assert validate_email("  dev@example.com ") == "dev@example.com"


def test_email_validator_invalid_cases() -> None:
    assert is_valid_email(None) is False
    assert is_valid_email("") is False
    assert is_valid_email("dev@localhost") is False
    assert is_valid_email("dev..x@example.com") is False
    assert is_valid_email(".dev@example.com") is False
    assert is_valid_email("dev.@example.com") is False

    with pytest.raises(ValidationError):
        validate_email(123)
    with pytest.raises(ValidationError):
        validate_email("not-an-email")


def test_url_validator_valid_cases() -> None:
    assert is_valid_url("https://example.com") is True
    assert is_valid_url("http://localhost:8080/path?q=1#frag") is True
    assert is_valid_url("https://127.0.0.1") is True
    assert validate_url(" https://example.com/a ") == "https://example.com/a"


def test_url_validator_invalid_cases() -> None:
    assert is_valid_url(None) is False
    assert is_valid_url("") is False
    assert is_valid_url("example.com") is False
    assert is_valid_url("ftp://example.com") is False
    assert is_valid_url("http://") is False
    assert is_valid_url("http://ex ample.com") is False
    assert is_valid_url("http://-bad.example.com") is False

    with pytest.raises(ValidationError):
        validate_url(123)
    with pytest.raises(ValidationError):
        validate_url("example.com")


def test_parse_date_accepts_multiple_formats() -> None:
    assert parse_date("2025-01-03") == date(2025, 1, 3)
    assert parse_date("2025/01/03") == date(2025, 1, 3)
    assert parse_date("20250103") == date(2025, 1, 3)
    assert parse_date("03-01-2025") == date(2025, 1, 3)
    assert parse_date(datetime(2025, 1, 3, 12, 0)) == date(2025, 1, 3)


def test_parse_date_rejects_invalid() -> None:
    assert is_valid_date("2025-01-03") is True
    assert is_valid_date("not-a-date") is False

    with pytest.raises(DateTimeValidationError):
        parse_date("")
    with pytest.raises(DateTimeValidationError):
        parse_date("2025-02-29")  # invalid (not a leap year)
    with pytest.raises(DateTimeValidationError):
        parse_date(123)


def test_parse_datetime_accepts_iso_and_common_formats() -> None:
    dtz = parse_datetime("2025-01-03T12:30:00Z")
    assert dtz.tzinfo is not None
    assert dtz.tzinfo.utcoffset(dtz) == timezone.utc.utcoffset(dtz)

    dto = parse_datetime("2025-01-03T12:30:00+02:00")
    assert dto.tzinfo is not None

    dt = parse_datetime("2025-01-03 12:30")
    assert dt.tzinfo is not None
    assert dt.hour == 12
    assert dt.minute == 30

    d = parse_datetime(date(2025, 1, 3))
    assert d.hour == 0 and d.minute == 0 and d.second == 0


def test_parse_datetime_rejects_invalid() -> None:
    assert is_valid_datetime("2025-01-03T12:30:00Z") is True
    assert is_valid_datetime("nope") is False

    with pytest.raises(DateTimeValidationError):
        parse_datetime("")
    with pytest.raises(DateTimeValidationError):
        parse_datetime("2025-13-01T00:00:00")
    with pytest.raises(DateTimeValidationError):
        parse_datetime(123)


def test_json_schema_object_required_properties_and_additional_properties() -> None:
    schema = {
        "type": "object",
        "required": ["id", "email"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "email": {"type": "string", "format": "email"},
        },
        "additionalProperties": False,
    }

    validate_json_schema({"id": "abc", "email": "dev@example.com"}, schema)

    with pytest.raises(SchemaValidationError, match="Missing required property"):
        validate_json_schema({"id": "abc"}, schema)
    with pytest.raises(SchemaValidationError, match="Additional property"):
        validate_json_schema({"id": "abc", "email": "dev@example.com", "extra": 1}, schema)


def test_json_schema_arrays_numbers_patterns_and_enum() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "pattern": r"^[a-z]+$"},
            },
            "age": {"type": "integer", "minimum": 0, "maximum": 130},
            "status": {"enum": ["new", "old"]},
        },
        "required": ["tags", "age", "status"],
    }

    validate_json_schema({"tags": ["abc"], "age": 30, "status": "new"}, schema)

    with pytest.raises(SchemaValidationError, match="pattern"):
        validate_json_schema({"tags": ["ABC"], "age": 30, "status": "new"}, schema)
    with pytest.raises(SchemaValidationError, match="minimum"):
        validate_json_schema({"tags": ["abc"], "age": -1, "status": "new"}, schema)
    with pytest.raises(SchemaValidationError, match="enum"):
        validate_json_schema({"tags": ["abc"], "age": 30, "status": "nope"}, schema)


def test_json_schema_anyof_allof_oneof() -> None:
    schema_anyof = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    validate_json_schema("x", schema_anyof)
    validate_json_schema(3, schema_anyof)
    with pytest.raises(SchemaValidationError, match="anyOf"):
        validate_json_schema([], schema_anyof)

    schema_allof = {"allOf": [{"type": "string", "minLength": 2}, {"type": "string", "maxLength": 3}]}
    validate_json_schema("ab", schema_allof)
    with pytest.raises(SchemaValidationError):
        validate_json_schema("a", schema_allof)

    schema_oneof = {"oneOf": [{"type": "integer"}, {"type": "number"}]}
    with pytest.raises(SchemaValidationError, match="oneOf"):
        validate_json_schema(1, schema_oneof)  # matches both integer and number
    validate_json_schema(1.5, schema_oneof)


def test_json_schema_format_uri_date_and_date_time() -> None:
    schema = {
        "type": "object",
        "properties": {
            "site": {"type": "string", "format": "uri"},
            "d": {"type": "string", "format": "date"},
            "dt": {"type": "string", "format": "date-time"},
        },
        "required": ["site", "d", "dt"],
    }
    validate_json_schema(
        {"site": "https://example.com", "d": "2025-01-03", "dt": "2025-01-03T12:00:00Z"},
        schema,
    )
    with pytest.raises(SchemaValidationError, match="URI"):
        validate_json_schema({"site": "not-a-url", "d": "2025-01-03", "dt": "2025-01-03T12:00:00Z"}, schema)


def test_validate_json_string_and_is_valid_json_schema() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    assert validate_json_string('{"x": 1}', schema) == {"x": 1}
    assert is_valid_json_schema({"x": 1}, schema) is True
    assert is_valid_json_schema({"x": "no"}, schema) is False

    with pytest.raises(SchemaValidationError, match="Invalid JSON"):
        validate_json_string("{", schema)


def test_json_schema_schema_shape_errors() -> None:
    with pytest.raises(SchemaValidationError, match="Schema must be a mapping"):
        validate_json_schema({"x": 1}, schema=123)  # type: ignore[arg-type]

    with pytest.raises(SchemaValidationError, match="required"):
        validate_json_schema({"x": 1}, schema={"type": "object", "required": "x"})  # type: ignore[arg-type]

    with pytest.raises(SchemaValidationError, match="properties"):
        validate_json_schema({"x": 1}, schema={"type": "object", "properties": 1})  # type: ignore[arg-type]

    with pytest.raises(SchemaValidationError, match="type"):
        validate_json_schema({"x": 1}, schema={"type": 5})  # type: ignore[arg-type]


def test_custom_validator_builder_composition() -> None:
    v = non_empty_string().then(matches_regex(r"^\d{4}$", "4 digits")).map(int)
    assert v("2025") == 2025
    with pytest.raises(ValidationError, match="4 digits"):
        v("x")

    v2 = compose(non_empty_string(), min_length(2), max_length(3))
    assert v2("ab") == "ab"
    with pytest.raises(ValidationError):
        v2("a")


def test_custom_validator_optional_or_else_one_of_range() -> None:
    v = one_of({"a", "b"}).optional()
    assert v(None) is None
    assert v("a") == "a"
    with pytest.raises(ValidationError):
        v("c")

    v3 = matches_regex(r"^\d+$", "digits").or_else(one_of({"x"}))
    assert v3("123") == "123"
    assert v3("x") == "x"
    with pytest.raises(ValidationError):
        v3("nope")

    assert in_range(0, 10)(3) == 3
    with pytest.raises(ValidationError, match="less than"):
        in_range(0, 10)(-1)

