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

from atomic_femdvr.atomic import AtomicInput, solve_atomic
from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic
from atomic_femdvr.utils import PlotWavefunctions, PrintEigenvalues, PrintTime

__all__ = [
    "atomic",
    "pseudoatomic"
]

@click.group()
def main() -> None:
    pass

# If you want to have a multi-command CLI, see https://click.palletsprojects.com/en/latest/commands/
@main.command()
@click.option("--plot", is_flag=True, help="Plot the results")
@click.argument("input_file", type=click.Path(exists=True))
def atomic(input_file: str, plot: bool) -> None:


    tic = perf_counter()

    # Read input parameters
    with open(input_file) as f:
        data = json.load(f)
    inp = AtomicInput(**data)

    eigenvalues, r_grid, psi = solve_atomic(inp.sysparams, inp.solver)

    PrintEigenvalues(inp.sysparams.lmax, eigenvalues)

    if plot:
        PlotWavefunctions(r_grid, psi, inp.sysparams.lmax, eigenvalues)

    toc = perf_counter()
    PrintTime(tic, toc, "Total")

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



if __name__ == "__main__":
    main()
