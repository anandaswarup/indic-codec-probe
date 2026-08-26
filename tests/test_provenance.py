from datetime import UTC, datetime

from indic_codec_probe.provenance import make_run_id, sha256_json


def test_sha256_json_is_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_make_run_id_binds_revisions() -> None:
    run_id = make_run_id(
        "a" * 40,
        "b" * 64,
        "c" * 64,
        now=datetime(2026, 8, 26, 12, 34, 56, tzinfo=UTC),
    )

    assert run_id == "20260826T123456Z-aaaaaaaa-bbbbbbbb-cccccccc"
