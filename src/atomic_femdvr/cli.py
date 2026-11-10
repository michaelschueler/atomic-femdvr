"""Command line interface for :mod:`atomic_femdvr`."""

import json

import click

from atomic_femdvr.full_atomic import TASKS as AE_TASKS
from atomic_femdvr.full_atomic import FullAtomicInput, solve_atomic
from atomic_femdvr.input import PhotoemissionInput
from atomic_femdvr.pseudo_atomic import TASKS as PSEUDO_TASKS
from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic
from atomic_femdvr.version import get_version

__all__ = [
    "atomic",
    "main",
    "photoemission",
    "pseudoatomic",
    "pseudophotoemission",
]


def _read_json(input_file: str) -> dict:
    with open(input_file) as f:
        return json.load(f)


@click.group()
@click.version_option(version=get_version(), message="atomic_femdvr %(version)s")
def main() -> None:
    """Kohn-Sham solver for isolated atoms (all-electron and pseudopotential) using FE-DVR."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-t", "--task", type=str, multiple=True, required=True,
              help=f"Task(s) to run: {', '.join(AE_TASKS)}.")
@click.option("--plot", is_flag=True, help="Plot the wavefunctions.")
@click.argument("export_dir", type=click.Path(), required=False)
def atomic(input_file: str, task: tuple[str, ...], plot: bool, export_dir: str | None) -> None:
    """All-electron calculation defined by INPUT_FILE (JSON)."""
    data = _read_json(input_file)
    data.pop('photoemission', None)
    solve_atomic(FullAtomicInput(**data), task, plot, export_dir)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-t", "--task", type=str, multiple=True, required=True,
              help=f"Task(s) to run, comma-separated or repeated: {', '.join(PSEUDO_TASKS)}.")
@click.option("--plot", is_flag=True, help="Plot the wavefunctions.")
@click.argument("export_dir", type=click.Path(), required=False)
def pseudoatomic(input_file: str, task: tuple[str, ...], plot: bool, export_dir: str | None) -> None:
    """Pseudo-atom calculation defined by INPUT_FILE (JSON) using a UPF pseudopotential.

    If EXPORT_DIR is given, the orbitals from the 'nscf' task are written there.
    """
    data = _read_json(input_file)
    data.pop('photoemission', None)
    solve_pseudo_atomic(PseudoAtomicInput(**data), task, plot, export_dir)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
def photoemission(input_file: str) -> None:
    """Compute photoemission matrix elements from a completed all-electron SCF calculation.

    INPUT_FILE is the same JSON as used for 'atomic', with an additional
    'photoemission' section specifying the energy range and output options.
    The KS potential must already be saved (run 'atomic -t scf' first).
    """
    from atomic_femdvr.photo_atomic import solve_photoemission_atomic

    data = _read_json(input_file)
    photo_inp = PhotoemissionInput(**data.pop('photoemission', {}))
    solve_photoemission_atomic(FullAtomicInput(**data), photo_inp)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
def pseudophotoemission(input_file: str) -> None:
    """Compute photoemission matrix elements from a completed pseudo-atomic SCF calculation.

    INPUT_FILE is the same JSON as used for 'pseudoatomic', with an additional
    'photoemission' section specifying the energy range and output options.
    The KS potential must already be saved (run 'pseudoatomic -t scf' first).
    """
    from atomic_femdvr.photo_pseudo import solve_photoemission_pseudo

    data = _read_json(input_file)
    photo_inp = PhotoemissionInput(**data.pop('photoemission', {}))
    solve_photoemission_pseudo(PseudoAtomicInput(**data), photo_inp)


if __name__ == "__main__":
    main()
