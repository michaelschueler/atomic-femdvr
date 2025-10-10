import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer


#==================================================================
class upf_class:
    """
    Class to interface with the UPF library for atomic calculations.
    """

    #-------------------------------------------------------------------
    def __init__(self, lib_path: str, extension: str = "so"):
        lib_so = os.path.join(lib_path, 'libupflib' + '.' + extension)
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
            ctypes.POINTER(ctypes.c_int),    # lmax
            ctypes.POINTER(ctypes.c_int),    # nwfc
            ctypes.POINTER(ctypes.c_int)     # nbeta
        ]

        self.lib.Get_UPF_GridInfo.restype = None
        self.lib.Get_UPF_GridInfo.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # mesh
            ctypes.POINTER(ctypes.c_double), # xmin
            ctypes.POINTER(ctypes.c_double), # rmax
            ctypes.POINTER(ctypes.c_double)  # dx
        ]

        self.lib.Get_UPF_Grid.restype = None
        self.lib.Get_UPF_Grid.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # mesh
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS")   # r
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
    #-------------------------------------------------------------------
    def Read_UPF(self, filename: str) -> int:
        """
        Read UPF file.
        """
        iflag = self.lib.Read_UPF(filename.encode('utf-8'))

        if iflag != -2:
            raise RuntimeError(f"Error reading UPF file '{filename}'. Error code: {iflag}")

        # Initialize UPF basic parameters
        self.zp = ctypes.c_double()
        self.etotps = ctypes.c_double()
        self.ecutwfc = ctypes.c_double()
        self.ecutrho = ctypes.c_double()
        self.lmax = ctypes.c_int()
        self.nwfc = ctypes.c_int()
        self.nbeta = ctypes.c_int()


        self.lib.Get_UPF_Basic(ctypes.byref(self.zp), ctypes.byref(self.etotps),
                                ctypes.byref(self.ecutwfc), ctypes.byref(self.ecutrho),
                                ctypes.byref(self.lmax), ctypes.byref(self.nwfc),
                                ctypes.byref(self.nbeta))

        self.zp = self.zp.value
        self.etotps = self.etotps.value
        self.ecutwfc = self.ecutwfc.value
        self.ecutrho = self.ecutrho.value
        self.lmax = self.lmax.value
        self.nwfc = self.nwfc.value
        self.nbeta = self.nbeta.value

        # Initialize UPF grid information
        self.mesh = ctypes.c_int()
        self.xmin = ctypes.c_double()
        self.rmax = ctypes.c_double()
        self.dx = ctypes.c_double()
        self.lib.Get_UPF_GridInfo(ctypes.byref(self.mesh), ctypes.byref(self.xmin),
                                  ctypes.byref(self.rmax), ctypes.byref(self.dx))
        self.mesh = self.mesh.value
        self.xmin = self.xmin.value
        self.rmax = self.rmax.value
        self.dx = self.dx.value


        # Initialize UPF grid
        self.r = np.zeros(self.mesh, dtype=np.float64, order='F')
        mesh_ctype = ctypes.c_int(self.mesh)
        self.lib.Get_UPF_Grid(mesh_ctype, self.r)


    #-------------------------------------------------------------------
    def ReadWavefunctions(self):

        # Initialize UPF pseudo wavefunctions
        self.nchi = np.zeros(self.nwfc, dtype=np.int32, order='F')
        self.lchi = np.zeros(self.nwfc, dtype=np.int32, order='F')
        self.oc = np.zeros(self.nwfc, dtype=np.float64, order='F')
        self.rcut_chi = np.zeros(self.nwfc, dtype=np.float64, order='F')
        self.chi = np.zeros([self.mesh, self.nwfc], dtype=np.float64, order='F')

        mesh_ctype = ctypes.c_int(self.mesh)
        nwfc_ctype = ctypes.c_int(self.nwfc)
        self.lib.Get_UPF_PseudoWfs(mesh_ctype, nwfc_ctype,
                                   self.nchi, self.lchi, self.oc, self.rcut_chi, self.chi)

        self.nnodes_chi = np.zeros(self.nwfc, dtype=int)

        lmax = np.amax(self.lchi)

        l_list = []
        for iwf in range(self.nwfc):
            l = self.lchi[iwf]
            if l in l_list:
                self.nnodes_chi[iwf] += 1
            l_list.append(l)

    #-------------------------------------------------------------------
    def Read_PP(self):

        # Initialize UPF pseudopotential information
        self.lloc = ctypes.c_int()
        self.lll = np.zeros(self.nbeta, dtype=np.int32, order='F')
        self.dion = np.zeros([self.nbeta, self.nbeta], dtype=np.float64, order='F')

        self.lib.Get_UPF_PPInfo(self.nbeta, ctypes.byref(self.lloc), self.lll, self.dion)

        self.lloc = self.lloc.value

        # Initialize UPF pseudopotential
        self.vloc = np.zeros(self.mesh, dtype=np.float64, order='F')
        self.kbeta = np.zeros(self.nbeta, dtype=np.int32, order='F')
        self.beta = np.zeros([self.mesh, self.nbeta], dtype=np.float64, order='F')

        mesh_ctype = ctypes.c_int(self.mesh)
        nbeta_ctype = ctypes.c_int(self.nbeta)
        self.lib.Get_UPF_PP(mesh_ctype, nbeta_ctype, self.vloc, self.kbeta, self.beta)

        self.kbeta_max = np.max(self.kbeta)
    #-------------------------------------------------------------------
    def GetChargeDensity(self):
        """
        Compute the charge density from the wavefunctions.
        """
        rho = np.zeros(self.mesh, dtype=np.float64)
        for iwf in range(self.nwfc):
            rho[1:] += self.oc[iwf] * np.abs(self.chi[1:, iwf])**2 / self.r[1:]**2
        rho[0] = rho[1]

        return rho
    #-------------------------------------------------------------------
