import pytest

def test_cli_help(script_runner):
    """Test `atomic_femdvr --help`."""
    result = script_runner.run(["atomic_femdvr", "--help"])
    assert result.success
    assert result.stdout.startswith("Usage:")
    assert ("atomic_femdvr [OPTIONS]") in result.stdout

def test_atomic_cli_help(script_runner):
    """Test `atomic_femdvr atomic --help`."""
    result = script_runner.run(["atomic_femdvr", "atomic", "--help"])
    assert result.success
    assert result.stdout.startswith("Usage:")
    assert ("atomic_femdvr atomic [OPTIONS]") in result.stdout

def test_pseudoatomic_cli_help(script_runner):
    """Test `atomic_femdvr pseudoatomic --help`."""
    result = script_runner.run(["atomic_femdvr", "pseudoatomic", "--help"])
    assert result.success
    assert result.stdout.startswith("Usage:")
    assert ("atomic_femdvr pseudoatomic [OPTIONS]") in result.stdout
