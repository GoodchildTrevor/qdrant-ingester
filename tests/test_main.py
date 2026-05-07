from qdrant_ingester.main import _merge_payload


def test_merge_payload_allows_extra_keys():
    base = {"name": "doc"}
    extra = {"author": "alice"}
    merged = _merge_payload(base, extra)

    assert merged["author"] == "alice"
    assert merged["name"] == "doc"


def test_merge_payload_rejects_reserved_keys():
    base = {"name": "doc"}
    extra = {"name": "override"}

    with pytest.raises(Exception) as excinfo:
        _merge_payload(base, extra)

    assert "extra_payload cannot override reserved keys" in str(excinfo.value)
