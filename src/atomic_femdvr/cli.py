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
from time import perf_counter

import click

from atomic_femdvr.full_atomic import FullAtomicInput, solve_atomic
from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic
from atomic_femdvr.version import get_version

from atomic_femdvr.solver_test import solver_test
from atomic_femdvr.wavefunction_test import wfc_test
from atomic_femdvr.vxc_test import vxc_benchmark

__all__ = [
    "atomic",
    "debug",
    "pseudoatomic"
]

@click.group()
@click.version_option(version=get_version(),message="atomic_femdvr %(version)s")
def main() -> None:
    pass

# If you want to have a multi-command CLI, see https://click.palletsprojects.com/en/latest/commands/
@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-t", "--task", type=str, multiple=True, required=True)
@click.option("--plot", is_flag=True, help="Plot the results")
def atomic(input_file: str, task: tuple[str, ...], plot: bool) -> None:


    tic = perf_counter()

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
def pseudoatomic(input_file: str, task: tuple[str, ...], plot: bool, export_dir: str | None) -> None:

    with open(input_file) as f:
        data = json.load(f)

    inp = PseudoAtomicInput(**data)

    solve_pseudo_atomic(inp, task, plot, export_dir)


@main.command()
@click.argument("file_rho", type=click.Path(exists=True))
@click.argument("file_vh", type=click.Path(exists=True))
@click.option("--plot", is_flag=True, help="Plot the results")
def debug(file_rho: str, file_vh: str, plot: bool) -> None:


    # wfc_test(plot)
    # hartree_test(plot)
    # hartree_benchmark(file_rho, file_vh)
    vxc_benchmark(file_rho, file_vh)


if __name__ == "__main__":
    main()
