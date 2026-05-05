import ctypes
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.ctypeslib import ndpointer


# ==================================================================
class UPFInterface:
    """
    Class to interface with the UPF library for atomic calculations.
    """

    # -------------------------------------------------------------------
    def __init__(self, lib_path: Path, extension: str = "so"):
        warnings.warn(
            "This UPFInterface is deprecated. Use atomic_femdvr.upf.UPFInterface instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        lib_so = lib_path / ("libupflib" + "." + extension)
        self.lib = ctypes.CDLL(lib_so)

        self.lib.Read_UPF.restype = ctypes.c_int
        self.lib.Read_UPF.argtypes = [
            ctypes.c_char_p  # filename
        ]

        self.lib.Get_UPF_Basic.restype = None
        self.lib.Get_UPF_Basic.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # zp
            ctypes.POINTER(ctypes.c_double),  # etotps
            ctypes.POINTER(ctypes.c_double),  # ecutwfc
            ctypes.POINTER(ctypes.c_double),  # ecutrho
            ctypes.POINTER(ctypes.c_int),  # lmax
            ctypes.POINTER(ctypes.c_int),  # nwfc
            ctypes.POINTER(ctypes.c_int),  # nbeta
        ]

        self.lib.Get_UPF_GridInfo.restype = None
        self.lib.Get_UPF_GridInfo.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # mesh
            ctypes.POINTER(ctypes.c_double),  # xmin
            ctypes.POINTER(ctypes.c_double),  # rmax
            ctypes.POINTER(ctypes.c_double),  # dx
        ]

        self.lib.Get_UPF_Grid.restype = None
        self.lib.Get_UPF_Grid.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # mesh
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # r
        ]

        self.lib.Get_UPF_PseudoWfs.restype = None
        self.lib.Get_UPF_PseudoWfs.argtypes = [
            ctypes.c_int,  # mesh
            ctypes.c_int,  # nwfc
            ndpointer(ctypes.c_int, flags="F_CONTIGUOUS"),  # nchi
            ndpointer(ctypes.c_int, flags="F_CONTIGUOUS"),  # lchi
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # oc
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # rcut_chi,
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # chi
        ]

        self.lib.Get_UPF_PPInfo.restype = None
        self.lib.Get_UPF_PPInfo.argtypes = [
            ctypes.c_int,  # nbeta
            ctypes.POINTER(ctypes.c_int),  # lloc
            ndpointer(ctypes.c_int, flags="F_CONTIGUOUS"),  # lll
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # dion
        ]

        self.lib.Get_UPF_PP.restype = None
        self.lib.Get_UPF_PP.argtypes = [
            ctypes.c_int,  # mesh
            ctypes.c_int,  # nbeta
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # vloc
            ndpointer(ctypes.c_int, flags="F_CONTIGUOUS"),  # kbeta
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # beta
        ]

        self.lib.Set_UPF_PPInfo.restype = None
        self.lib.Set_UPF_PPInfo.argtypes = [
            ctypes.c_int,  # nbeta
            ctypes.c_int,  # lloc
            ndpointer(ctypes.c_int, flags="F_CONTIGUOUS"),  # lll
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # dion
        ]

        self.lib.Set_UPF_PP.restype = None
        self.lib.Set_UPF_PP.argtypes = [
            ctypes.c_int,  # mesh
            ctypes.c_int,  # nbeta
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # vloc
            ndpointer(ctypes.c_int, flags="F_CONTIGUOUS"),  # kbeta
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),  # beta
        ]

        self.lib.Save_UPF.restype = None
        self.lib.Save_UPF.argtypes = [
            ctypes.c_char_p  # filename
        ]

    # -------------------------------------------------------------------
    def Read_UPF(self, filename: Path) -> None:
        """
        Read UPF file.
        """
        iflag = self.lib.Read_UPF(str(filename).encode("utf-8"))

        if iflag != -2:
            raise RuntimeError(f"Error reading UPF file '{filename}'. Error code: {iflag}")

        # Initialize UPF basic parameters
        self._zp = ctypes.c_double()
        self._etotps = ctypes.c_double()
        self._ecutwfc = ctypes.c_double()
        self._ecutrho = ctypes.c_double()
        self._lmax = ctypes.c_int()
        self._nwfc = ctypes.c_int()
        self._nbeta = ctypes.c_int()

        self.lib.Get_UPF_Basic(
            ctypes.byref(self._zp),
            ctypes.byref(self._etotps),
            ctypes.byref(self._ecutwfc),
            ctypes.byref(self._ecutrho),
            ctypes.byref(self._lmax),
            ctypes.byref(self._nwfc),
            ctypes.byref(self._nbeta),
        )

        # Initialize UPF grid information
        self._mesh = ctypes.c_int()
        self._xmin = ctypes.c_double()
        self._rmax = ctypes.c_double()
        self._dx = ctypes.c_double()
        self.lib.Get_UPF_GridInfo(
            ctypes.byref(self._mesh),
            ctypes.byref(self._xmin),
            ctypes.byref(self._rmax),
            ctypes.byref(self._dx),
        )

        # Initialize UPF grid
        self.r = np.zeros(self.mesh, dtype=np.float64, order="F")
        mesh_ctype = ctypes.c_int(self.mesh)
        self.lib.Get_UPF_Grid(mesh_ctype, self.r)

    @property
    def zp(self) -> float:
        return self._zp.value

    @property
    def etotps(self) -> float:
        return self._etotps.value

    @property
    def ecutwfc(self) -> float:
        return self._ecutwfc.value

    @property
    def ecutrho(self) -> float:
        return self._ecutrho.value

    @property
    def lmax(self) -> int:
        return self._lmax.value

    @property
    def nwfc(self) -> int:
        return self._nwfc.value

    @property
    def nbeta(self) -> int:
        return self._nbeta.value

    @property
    def mesh(self) -> int:
        return self._mesh.value

    @property
    def xmin(self) -> float:
        return self._xmin.value

    @property
    def rmax(self) -> float:
        return self._rmax.value

    @property
    def dx(self) -> float:
        return self._dx.value

    @property
    def lloc(self) -> int:
        return self._lloc.value

    # -------------------------------------------------------------------
    def ReadWavefunctions(self) -> None:
        # Initialize UPF pseudo wavefunctions
        self.nchi = np.zeros(self.nwfc, dtype=np.int32, order="F")
        self.lchi = np.zeros(self.nwfc, dtype=np.int32, order="F")
        self.oc = np.zeros(self.nwfc, dtype=np.float64, order="F")
        self.rcut_chi = np.zeros(self.nwfc, dtype=np.float64, order="F")
        self.chi = np.zeros([self.mesh, self.nwfc], dtype=np.float64, order="F")

        mesh_ctype = ctypes.c_int(self.mesh)
        nwfc_ctype = ctypes.c_int(self.nwfc)
        self.lib.Get_UPF_PseudoWfs(
            mesh_ctype, nwfc_ctype, self.nchi, self.lchi, self.oc, self.rcut_chi, self.chi
        )

        self.nnodes_chi = np.zeros(self.nwfc, dtype=int)

        np.amax(self.lchi)

        l_list = []
        for iwf in range(self.nwfc):
            l = self.lchi[iwf]
            if l in l_list:
                self.nnodes_chi[iwf] += 1
            l_list.append(l)

    # -------------------------------------------------------------------
    def Read_PP(self) -> None:
        # Initialize UPF pseudopotential information
        self._lloc = ctypes.c_int()
        self.lll = np.zeros(self.nbeta, dtype=np.int32, order="F")
        self.dion = np.zeros([self.nbeta, self.nbeta], dtype=np.float64, order="F")

        self.lib.Get_UPF_PPInfo(self.nbeta, ctypes.byref(self._lloc), self.lll, self.dion)

        # Initialize UPF pseudopotential
        self.vloc = np.zeros(self.mesh, dtype=np.float64, order="F")
        self.kbeta = np.zeros(self.nbeta, dtype=np.int32, order="F")
        self.beta = np.zeros([self.mesh, self.nbeta], dtype=np.float64, order="F")

        mesh_ctype = ctypes.c_int(self.mesh)
        nbeta_ctype = ctypes.c_int(self.nbeta)
        self.lib.Get_UPF_PP(mesh_ctype, nbeta_ctype, self.vloc, self.kbeta, self.beta)

        self.kbeta_max = np.max(self.kbeta)

    # -------------------------------------------------------------------
    def GetChargeDensity(self) -> npt.NDArray[np.float64]:
        """
        Compute the charge density from the wavefunctions.
        """
        rho = np.zeros(self.mesh, dtype=np.float64)
        for iwf in range(self.nwfc):
            rho[1:] += self.oc[iwf] * np.abs(self.chi[1:, iwf]) ** 2 / self.r[1:] ** 2
        rho[0] = rho[1]

        return rho

    # -------------------------------------------------------------------
