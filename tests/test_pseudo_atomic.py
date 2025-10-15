import pytest

from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic


def test_Mo():
    inp = PseudoAtomicInput(
        pseudo_config={
            "upflib_dir": "/home/linsco_e/code/schueler/qe_upflib/lib",
            "lib_ext": "so",
            "storage_dir": "./Mo_Pseudo"
        },
        sysparams={
            "file_upf": "/home/linsco_e/code/koopmans/src/koopmans/pseudopotentials/PseudoDojo/0.4/PBE/SR/standard/upf/Mo.upf",
            "element": "Mo",
            "lmax": 3,
            "nmax": 2
        },
        solver={
            "h_min": 0.5,
            "h_max": 4.0,
            "Rmax": 30.0,
            "elem_tol": 1.0e-2,
            "ng": 8
        },
        dft={
            "driver": "internal",
            "x_functional": "GGA_X_PBE",
            "c_functional": "GGA_C_PBE",
            "alpha_x": 1.0,
            "max_iter": 100,
            "conv_tol": 1.0e-6
        },
        confinement={
            "type": "SoftStep",
            "rc": 10.0,
            "ri_factor": 0.5,
            "Vbarrier": 1.0,
            "polarization_mode": "SoftCoul",
            "softcoul_delta": 0.1
        },
        projector={
            "nr": 1001,
            "rmin": 1.0e-8
        }
    )

    eigenvalues = solve_pseudo_atomic(inp, task_list=('scf',))

    assert 'scf' in eigenvalues

    benchmark_eigenvalues = {'0': [-2.3767770200534524, -0.1603849407697104],
                             '1': [-1.434869395676964, -0.04179242855203131],
                             '2': [-0.1653870399482107]}
    
    for l in benchmark_eigenvalues:
        assert l in eigenvalues['scf']
        for ev, bev in zip(eigenvalues['scf'][l], benchmark_eigenvalues[l], strict=True):
            assert pytest.approx(ev, rel=1.0e-7) == bev




