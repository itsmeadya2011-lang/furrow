import json

import pytest

from furrow.util import parse_json_response


def test_parse_plain_json():
    assert parse_json_response('{"passed": true}') == {"passed": True}


def test_parse_strips_markdown_fence():
    response = '```json\n{"passed": true}\n```'
    assert parse_json_response(response) == {"passed": True}


def test_parse_strips_surrounding_prose():
    response = 'Here is the result:\n{"passed": false, "summary": "oops"}\nDone.'
    assert parse_json_response(response) == {"passed": False, "summary": "oops"}


def test_parse_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json_response("not json")
