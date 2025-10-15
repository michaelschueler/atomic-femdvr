import pytest

def test_cli_help(script_runner):
    """Test `atomic_femdvr --help`."""
    result = script_runner.run(["atomic_femdvr", "--help"])
    assert result.success
    assert result.stdout.startswith("Usage:")
    assert ("atomic_femdvr [OPTIONS]") in result.stdout
