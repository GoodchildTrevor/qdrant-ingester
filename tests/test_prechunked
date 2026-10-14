import json

import pytest

from qdrant_ingester.prechunked import (
    PrechunkedParseError,
    is_prechunked,
    parse_prechunked_file,
)

# Example chunk from an ERP export of Oracle PL/SQL objects
ERP_CHUNK = {
    "chunk_id": "standalone_function.AAA_CALC_SUM_FACT_PAY.acf2a0959bb9",
    "package_name": None,
    "procedure_name": "AAA_CALC_SUM_FACT_PAY",
    "kind": "standalone_function",
    "content": "CREATE OR REPLACE FUNCTION AAA_CALC_SUM_FACT_PAY ...",
    "signature": None,
    "start_line": 1,
    "end_line": 16,
    "source_file": "erp_export/functions/AAA_CALC_SUM_FACT_PAY.sql",
    "file_size": 425,
}


def test_is_prechunked_by_extension(tmp_path):
    assert is_prechunked(tmp_path / "a.jsonl") is True
    assert is_prechunked(tmp_path / "a.ndjson") is True
    assert is_prechunked(tmp_path / "a.JSON") is True
    assert is_prechunked(tmp_path / "a.pdf") is False
    assert is_prechunked(tmp_path / "a.txt") is False


def test_is_prechunked_explicit_override(tmp_path):
    assert is_prechunked(tmp_path / "a.pdf", override=True) is True
    assert is_prechunked(tmp_path / "a.jsonl", override=False) is False


def test_parse_jsonl(tmp_path):
    path = tmp_path / "chunks.jsonl"
    second = {**ERP_CHUNK, "chunk_id": "other", "content": "PROCEDURE BBB"}
    path.write_text(
        json.dumps(ERP_CHUNK) + "\n\n" + json.dumps(second), encoding="utf-8"
    )

    resp = parse_prechunked_file(path)

    assert resp.file_name == "chunks.jsonl"
    assert resp.file_format == "prechunked/jsonl"
    assert resp.creation_date and resp.modification_date
    assert len(resp.chunks) == 2

    first = resp.chunks[0]
    assert first.raw == ERP_CHUNK["content"]
    assert first.lemmas == ERP_CHUNK["content"]  # no lemmas -> fallback to content
    # all non-text fields are preserved as metadata
    assert first.meta["chunk_id"] == ERP_CHUNK["chunk_id"]
    assert first.meta["procedure_name"] == "AAA_CALC_SUM_FACT_PAY"
    assert first.meta["kind"] == "standalone_function"
    assert first.meta["start_line"] == 1
    assert first.meta["source_file"] == "erp_export/functions/AAA_CALC_SUM_FACT_PAY.sql"
    assert "content" not in first.meta
    assert "lemmas" not in first.meta

    assert resp.chunks[1].raw == "PROCEDURE BBB"


def test_parse_jsonl_with_lemmas(tmp_path):
    path = tmp_path / "lemmas.jsonl"
    path.write_text(
        json.dumps({"content": "CREATE FUNCTION X", "lemmas": "create function x"}),
        encoding="utf-8",
    )
    chunk = parse_prechunked_file(path).chunks[0]
    assert chunk.raw == "CREATE FUNCTION X"
    assert chunk.lemmas == "create function x"


def test_parse_json_array(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps([{"content": "one"}, {"content": "two"}]), encoding="utf-8"
    )
    resp = parse_prechunked_file(path)
    assert [c.raw for c in resp.chunks] == ["one", "two"]
    assert resp.file_format == "prechunked/json"


def test_parse_json_chunks_wrapper_and_single_object(tmp_path):
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"chunks": [{"raw": "text-a"}, {"text": "text-b"}]}),
        encoding="utf-8",
    )
    resp = parse_prechunked_file(wrapped)
    assert [c.raw for c in resp.chunks] == ["text-a", "text-b"]

    single = tmp_path / "single.json"
    single.write_text(json.dumps({"content": "solo"}), encoding="utf-8")
    assert parse_prechunked_file(single).chunks[0].raw == "solo"


def test_parse_jsonl_file_holding_a_json_array(tmp_path):
    # sloppy exporter wrote an array into a .jsonl file
    path = tmp_path / "sloppy.jsonl"
    path.write_text(
        "[\n  {\"content\": \"a\"},\n  {\"content\": \"b\"}\n]", encoding="utf-8"
    )
    assert [c.raw for c in parse_prechunked_file(path).chunks] == ["a", "b"]


def test_parse_errors(tmp_path):
    bad_line = tmp_path / "bad.jsonl"
    bad_line.write_text('{"content": "ok"}\n{not json}', encoding="utf-8")
    with pytest.raises(PrechunkedParseError, match="line 2"):
        parse_prechunked_file(bad_line)

    no_text = tmp_path / "notext.jsonl"
    no_text.write_text(json.dumps({"chunk_id": "x"}), encoding="utf-8")
    with pytest.raises(PrechunkedParseError, match="text field"):
        parse_prechunked_file(no_text)

    not_object = tmp_path / "scalar.jsonl"
    not_object.write_text('"just a string"', encoding="utf-8")
    with pytest.raises(PrechunkedParseError, match="not a JSON object"):
        parse_prechunked_file(not_object)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n \n", encoding="utf-8")
    with pytest.raises(PrechunkedParseError, match="no chunks"):
        parse_prechunked_file(empty)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[{", encoding="utf-8")
    with pytest.raises(PrechunkedParseError, match="invalid JSON"):
        parse_prechunked_file(bad_json)


def test_parse_non_utf8(tmp_path):
    path = tmp_path / "binary.jsonl"
    path.write_bytes(b'{"content": "\xff\xfe"}')
    with pytest.raises(PrechunkedParseError, match="UTF-8"):
        parse_prechunked_file(path)


# Example chunk from a Hyperion Plan HP export: the text lives inside "payload"
HP_CHUNK = {
    "id": 6656971675031269153,
    "payload": {
        "type": "dimension_definition",
        "app": "ECG_C0st",
        "db": "Plan1",
        "dimension": "ECG_CostAccount",
        "source_path": "./HP_export\\ECG_CostAccount",
        "fields": {"DimensionClass": "Account"},
        "text": "Измерение: ECG_CostAccount (приложение ECG_C0st.Plan1, файл ECG_CostAccount)",
    },
}


def test_parse_jsonl_nested_payload(tmp_path):
    path = tmp_path / "hp_export_chunks.jsonl"
    path.write_text(json.dumps(HP_CHUNK, ensure_ascii=False), encoding="utf-8")
    chunk = parse_prechunked_file(path).chunks[0]

    assert chunk.raw == HP_CHUNK["payload"]["text"]
    # no lemmas anywhere -> embedding falls back to the nested text
    assert chunk.lemmas == HP_CHUNK["payload"]["text"]
    # the top-level "id" and the payload metadata are preserved
    assert chunk.meta["id"] == 6656971675031269153
    assert chunk.meta["type"] == "dimension_definition"
    assert chunk.meta["app"] == "ECG_C0st"
    assert chunk.meta["db"] == "Plan1"
    assert chunk.meta["dimension"] == "ECG_CostAccount"
    assert chunk.meta["fields"] == {"DimensionClass": "Account"}
    # neither text keys nor lemmas leak into metadata, wrapper is unwrapped
    assert "text" not in chunk.meta
    assert "content" not in chunk.meta
    assert "raw" not in chunk.meta
    assert "lemmas" not in chunk.meta
    assert "payload" not in chunk.meta


def test_top_level_text_beats_nested_payload(tmp_path):
    obj = {
        "id": 1,
        "content": "top-level text",
        "payload": {"text": "nested text", "app": "ECG_C0st"},
    }
    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps(obj), encoding="utf-8")
    chunk = parse_prechunked_file(path).chunks[0]

    # pre-change behaviour: the top-level text wins, no unwrap happens
    assert chunk.raw == "top-level text"
    assert chunk.lemmas == "top-level text"
    assert chunk.meta["id"] == 1
    assert chunk.meta["payload"] == {"text": "nested text", "app": "ECG_C0st"}
    assert "content" not in chunk.meta


def test_nested_lemmas_used_for_embedding(tmp_path):
    obj = {"id": 2, "payload": {"text": "raw sentence", "lemmas": "lemma sentenc"}}
    path = tmp_path / "lemmas_nested.jsonl"
    path.write_text(json.dumps(obj), encoding="utf-8")
    chunk = parse_prechunked_file(path).chunks[0]

    assert chunk.raw == "raw sentence"
    assert chunk.lemmas == "lemma sentenc"
    assert chunk.lemmas != chunk.raw  # lemmas really used for embedding
    assert "lemmas" not in chunk.meta


def test_top_level_lemmas_beat_nested_lemmas(tmp_path):
    obj = {
        "id": 3,
        "lemmas": "top lemma",
        "payload": {"text": "raw", "lemmas": "nested lemma"},
    }
    path = tmp_path / "lemmas_conflict.jsonl"
    path.write_text(json.dumps(obj), encoding="utf-8")
    chunk = parse_prechunked_file(path).chunks[0]

    assert chunk.raw == "raw"
    assert chunk.lemmas == "top lemma"
    assert "lemmas" not in chunk.meta


def test_no_text_on_top_level_nor_in_payload(tmp_path):
    obj = {"id": 4, "payload": {"app": "ECG_C0st", "type": "member"}}
    path = tmp_path / "notext_nested.jsonl"
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(PrechunkedParseError, match=r"top level.*payload"):
        parse_prechunked_file(path)


def test_meta_key_conflict_top_level_wins(tmp_path):
    obj = {"id": 5, "type": "top", "payload": {"text": "raw", "type": "nested"}}
    path = tmp_path / "conflict.jsonl"
    path.write_text(json.dumps(obj), encoding="utf-8")
    chunk = parse_prechunked_file(path).chunks[0]
    assert chunk.raw == "raw"
    assert chunk.meta["type"] == "top"
