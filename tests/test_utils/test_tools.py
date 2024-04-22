import os
import pytest
from yaml import YAMLError
from unittest.mock import mock_open, patch

import sys

sys.path.append("./")
from utils.tools import complete_path, load_model_cfg


# Test for complete_path function
def test_complete_path():
    assert complete_path() == "./config/model/v7-base.yaml"

    assert complete_path("test") == "./config/model/test.yaml"

    assert complete_path("test.yaml") == "./config/model/test.yaml"


# Test for load_model_cfg function
def test_load_model_cfg_success():
    test_yaml_content = """
    nc: 80
    model:
        type: "yolo"
    """
    with patch("builtins.open", mock_open(read_data=test_yaml_content)):
        with patch("os.path.exists", return_value=True):
            result = load_model_cfg("v7-base.yaml")
            assert result["nc"] == 80
            assert result["model"]["type"] == "yolo"


def test_load_model_cfg_file_not_found():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_model_cfg("missing-file.yaml")


def test_load_model_cfg_yaml_error():
    # Simulating a YAML error
    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = YAMLError("error parsing YAML")
        with pytest.raises(YAMLError):
            load_model_cfg("corrupt-file.yaml")


def test_load_model_cfg_missing_keys():
    test_yaml_content = """
    nc: 80
    """
    with patch("builtins.open", mock_open(read_data=test_yaml_content)):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(ValueError) as exc_info:
                load_model_cfg("incomplete-model.yaml")
            assert str(exc_info.value) == "Missing required key: 'model'"
