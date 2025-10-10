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

from time import perf_counter

import click

from atomic_femdvr.atomic import ReadInput, SolveAtomic
from atomic_femdvr.utils import PlotWavefunctions, PrintEigenvalues, PrintTime

__all__ = [
    "main",
]


# If you want to have a multi-command CLI, see https://click.palletsprojects.com/en/latest/commands/
@click.command()
@click.option("--plot", is_flag=True, help="Plot the results")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(), required=False)
def main(input_file: str, output_file: str | None, plot: bool) -> None:


    tic = perf_counter()

    # Read input parameters
    sysparams, solver = ReadInput(input_file)

    eigenvalues, r_grid, psi = SolveAtomic(sysparams, solver)

    lmax = sysparams.get('lmax', 0)
    PrintEigenvalues(lmax, eigenvalues)

    if plot:
        PlotWavefunctions(r_grid, psi, lmax, eigenvalues)

    toc = perf_counter()
    PrintTime(tic, toc, "Total")


if __name__ == "__main__":
    main()
