"""Command line interface for :mod:`atomic_femdvr`.

Why does this file exist, and why not put this in ``__main__``?
You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m atomic_femdvr`` python will
  execute``__main__.py`` as a script. That means there won't be any
  ``atomic_femdvr.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``atomic_femdvr.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/8.1.x/setuptools/#setuptools-integration
"""

import json
import logging
from time import perf_counter

import click

from atomic_femdvr.full_atomic import FullAtomicInput, solve_atomic
from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic
from atomic_femdvr.version import get_version

__all__ = [
    "atomic",
    "pseudoatomic",
]


def _setup_logging(verbose: bool) -> None:
    """Configure the root atomic_femdvr logger so the CLI prints progress.

    The package modules log their progress via ``logger.info`` / ``warning``;
    library callers can configure their own handlers, but the CLI installs a
    minimal stdout handler so the user sees output by default.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG-level logging.")
@click.version_option(version=get_version(), message="atomic_femdvr %(version)s")
def main(verbose: bool) -> None:
    _setup_logging(verbose)


# If you want to have a multi-command CLI, see https://click.palletsprojects.com/en/latest/commands/
@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-t", "--task", type=str, multiple=True, required=True)
@click.option("--plot", is_flag=True, help="Plot the results")
def atomic(input_file: str, task: tuple[str, ...], plot: bool) -> None:
    perf_counter()

    # Read input parameters
    with open(input_file) as f:
        data = json.load(f)
    inp = FullAtomicInput(**data)

    solve_atomic(inp, task, plot)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-t", "--task", type=str, multiple=True, required=True)
@click.option("--plot", is_flag=True, help="Plot the results")
@click.argument("export_dir", type=click.Path(), required=False)
def pseudoatomic(
    input_file: str, task: tuple[str, ...], plot: bool, export_dir: str | None
) -> None:
    with open(input_file) as f:
        data = json.load(f)

    inp = PseudoAtomicInput(**data)

    solve_pseudo_atomic(inp, task, plot, export_dir)


if __name__ == "__main__":
    main()
