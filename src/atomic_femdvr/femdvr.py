import numpy as np

from atomic_femdvr.legendre import Legendre
from atomic_femdvr.legendre_integrals import get_legendre_integrals


#=================================================================
class FEDVR_Basis:
	"""docstring for FEDVR_Basis"""

	def __init__(self, ne: int, ng: int, xp: list, build_derivatives: bool = True,
			   build_integrals: bool = False):
		self.ne = ne
		self.ng = ng
		self.xp = np.array(xp)
		self.leg = Legendre(ng+1)

		self.tts = np.zeros([ng+1,ng+1,ne])
		self.dts = np.zeros([ng+1,ng+1,ne])
		for i1 in range(self.ne):
			self.tts[:,:,i1] = ttilde(self.leg,self.xp[i1+1]-self.xp[i1])
			self.dts[:,:,i1] = dtilde(self.leg,self.xp[i1+1]-self.xp[i1])

		self.have_derivatives = build_derivatives
		if self.have_derivatives:
			self.der1_full = self.get_deriv_matrix_full(n=1, cplx=False)
			self.der2_full = self.get_deriv_matrix_full(n=2, cplx=False)

		self.have_integrals = build_integrals
		if self.have_integrals:
			self.leg_integ = get_legendre_integrals(self.leg, self.xp)
	#------------------------------------------------------------
	def get_gridpoints(self) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		xp = self.xp
		xg = np.zeros(ne*ng+1)
		for i in range(ne):
			xg[i*ng:i*ng+ng] = xp[i] + 0.5 * (xp[i+1] - xp[i]) * (self.leg.x_i[0:ng] + 1)
		xg[-1] = xp[-1]

		return xg
	#------------------------------------------------------------
	def get_psi_all(self, cff: np.ndarray, cplx: bool = False) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		nvec = cff.shape[1]

		if cplx:
			psi = np.zeros([ne*ng+1, nvec], dtype=np.complex128)
		else:
			psi = np.zeros([ne*ng+1, nvec])

		for i in range(nvec):
			psi[:,i] = self.get_psi(cff[:,i], cplx=cplx)
		return psi
	#------------------------------------------------------------
	def get_psi(self, cff: np.ndarray, cplx: bool = False) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		xp = self.xp

		cff_ext = np.concatenate((cff,[0.0]))
		v = np.reshape(cff_ext,[ne,ng])
		v = np.flip(v,axis=1)

		if cplx:
			psi = np.zeros(ne*ng+1, dtype=np.complex128)
		else:
			psi = np.zeros(ne*ng+1)

		for i in range(1,ne):
			w1 = 0.5 * (xp[i] - xp[i-1]) * self.leg.w_i[ng]
			w2 = 0.5 * (xp[i+1] - xp[i]) * self.leg.w_i[0]
			psi[i*ng] = v[i-1,0] / np.sqrt(w1 + w2)

		# for i in range(ne-1):
		# 	w1 = 0.5 * (xp[i+1] - xp[i]) * self.leg.w_i[ng]
		# 	w2 = 0.5 * (xp[i+2] - xp[i+1]) * self.leg.w_i[0]
		# 	psi[i*ng] = v[i + 1,0] / np.sqrt(w1 + w2)

		for i in range(ne):
			for m in range(1,ng):
				w = 0.5 * (xp[i+1] - xp[i]) * self.leg.w_i[m]
				psi[i*ng+m] = v[i,m] / np.sqrt(w)

		return psi
	#------------------------------------------------------------
	def get_overlap(self, psi: np.ndarray, phi: np.ndarray) -> float:

		cff_psi = self.get_coeffs(psi, cplx=False)
		cff_phi = self.get_coeffs(phi, cplx=False)

		overlap = np.dot(cff_psi, cff_phi)

		return overlap
	#------------------------------------------------------------
	def get_coeffs(self, psi: np.ndarray, cplx: bool = False) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		nb = ne*ng - 1
		xp = self.xp

		if cplx:
			cff4 = np.zeros([ne,ng], dtype=complex)
		else:
			cff4 = np.zeros([ne,ng])

		for i in range(1,ne):
			w1 = 0.5 * (xp[i] - xp[i-1]) * self.leg.w_i[ng]
			w2 = 0.5 * (xp[i+1] - xp[i]) * self.leg.w_i[0]
			cff4[i-1,0] = psi[i*ng] * np.sqrt(w1 + w2)

		for i in range(ne):
			for m in range(1,ng):
				w = 0.5 * (xp[i+1] - xp[i]) * self.leg.w_i[m]
				cff4[i,m] = psi[i*ng+m] * np.sqrt(w)

		cff2 = np.reshape(np.flip(cff4, axis=1), [ne*ng])

		return cff2[0:nb]
	#------------------------------------------------------------
	def to_linear_grid(self, psi: np.ndarray, nx: int) -> tuple[np.ndarray, np.ndarray]:

		ne = self.ne
		ng = self.ng

		grid = []
		psi_grid = []
		for i in range(ne):
			psi_elem = psi[i*ng : i*ng + ng + 1]
			c_elem = self.leg.to_spectral(psi_elem)

			xs = np.linspace(self.xp[i], self.xp[i+1], nx)
			xred = np.linspace(-1.0, 1.0, nx)
			psi_elem_lin = np.polynomial.legendre.legval(xred[0:nx-1], c_elem)
			psi_grid.append(psi_elem_lin)
			grid.append(xs[0:nx-1])

		psi_grid = np.concatenate(psi_grid)
		psi_grid = np.concatenate((psi_grid, [0.0]))  # Append zero for the last point
		grid = np.concatenate(grid)
		grid = np.concatenate((grid, [self.xp[-1]]))

		return grid, psi_grid
	#------------------------------------------------------------
	def interpolate(self, psi: np.ndarray, x: np.ndarray) -> np.ndarray:
		ne = self.ne
		ng = self.ng

		# Interpolate to the new grid points
		psi_interp = np.zeros_like(x, dtype=psi.dtype)
		for i in range(ne):
			Ix, = np.where((x >= self.xp[i]) & (x < self.xp[i+1]))
			if len(Ix) == 0:
				continue

			psi_elem = psi[i*ng : i*ng + ng + 1]
			c_elem = self.leg.to_spectral(psi_elem)

			# Normalize the x values to the range [-1, 1]
			t = -1.0 + 2.0 * (x[Ix] - self.xp[i]) / (self.xp[i+1] - self.xp[i])

			# Evaluate the polynomial at the normalized x values
			psi_interp[Ix] = np.polynomial.legendre.legval(t, c_elem)

		return psi_interp
	#------------------------------------------------------------
	def get_deriv_matrix_full(self, n: int = 2, cplx: bool = False) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		xp = self.xp
		nb = ne*ng-1
		w_i = self.leg.w_i

		if cplx:
			T4 = np.zeros([ne,ng,ne,ng],dtype=np.complex128)
		else:
			T4 = np.zeros([ne,ng,ne,ng])

		if n == 1:
			de = self.dts
		elif n == 2:
			de = 0.5 * self.tts
		else:
			raise ValueError("n must be 1 or 2 for first or second derivative")

		for i1 in range(ne):
			Li = xp[i1+1]-xp[i1]
			T4[i1,1:ng,i1,1:ng] = de[1:ng,1:ng,i1] / (0.5*Li * np.sqrt(w_i[1:ng,None]*w_i[None,1:ng]))

		for i1 in range(ne-1):
			w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i[ng] + 0.5*(xp[i1+2]-xp[i1+1]) * w_i[0]
			w2 = 0.5*(xp[i1+1]-xp[i1]) * w_i
			T4[i1,0,i1,1:ng] = de[-1,1:ng,i1]/np.sqrt(w1*w2[1:ng])

			i2 = i1 + 1
			w2 = 0.5*(xp[i2+1]-xp[i2]) * w_i
			T4[i1, 0, i2, 1:ng] = de[0, 1:ng, i2] / np.sqrt(w1 * w2[1:ng])

		for i2 in range(ne-1):
			w2 = 0.5*(xp[i2+1]-xp[i2]) * w_i[-1] + 0.5*(xp[i2+2]-xp[i2+1]) * w_i[0]

			i1 = i2
			w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i
			T4[i1, 1:ng, i2, 0] = de[1:ng, -1, i1] / np.sqrt(w1[1:ng] * w2)

			i1 = i2 + 1
			if i1 <= ne-1:
				w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i
				T4[i1, 1:ng, i2, 0] = de[1:ng, 0, i1] / np.sqrt(w1[1:ng] * w2)

		for i1 in range(ne-1):
			i2 = i1
			w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i[ng] + 0.5*(xp[i1+2]-xp[i1+1]) * w_i[0]
			w2 = w1
			T4[i1,0,i2,0] = (de[ng,ng,i1] + de[0,0,i1+1])/np.sqrt(w1*w2)

			i2 = i1+1
			if i2 <= ne-2:
				w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i[ng] + 0.5*(xp[i1+2]-xp[i1+1]) * w_i[0]
				w2 = 0.5*(xp[i2+1]-xp[i2]) * w_i[ng] + 0.5*(xp[i2+2]-xp[i2+1]) * w_i[0]
				T4[i1,0,i2,0] = de[0,ng,i2] / np.sqrt(w1*w2)

			i2 = i1 - 1
			if i2 >= 0:
				w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i[ng] + 0.5*(xp[i1+2]-xp[i1+1]) * w_i[0]
				w2 = 0.5*(xp[i2+1]-xp[i2]) * w_i[ng] + 0.5*(xp[i2+2]-xp[i2+1]) * w_i[0]
				T4[i1,0,i2,0] = de[ng,0,i1] / 	np.sqrt(w1*w2)

		return T4
	#------------------------------------------------------------
	def get_potential_from_func(self, Vfunc: callable) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		xp = self.xp
		nb = ne*ng-1
		w_i = self.leg.w_i
		x_i = self.leg.x_i

		Vvec = np.zeros(nb)
		V4 = np.zeros([ne,ng])

		for i1 in range(ne-1):
			w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i[ng]
			w2 = 0.5*(xp[i1+2]-xp[i1+1]) * w_i[0]
			xs1 = xp[i1] + 0.5 * (xp[i1+1] - xp[i1]) * (x_i[0:ng] + 1)
			xs2 = xp[i1+1] + 0.5 * (xp[i1+2] - xp[i1+1]) * (x_i[0:ng] + 1)

			V4[i1,0] = (Vfunc(xs1[-1]) * w1 + Vfunc(xs2[0]) * w2) / (w1 + w2)

		for i1 in range(ne):
			xs = xp[i1] + 0.5 * (xp[i1+1] - xp[i1]) * (x_i[0:ng] + 1)
			V4[i1,1:] = Vfunc(xs[1:])

		V2 = np.reshape(np.flip(V4, axis=1), [ne*ng])
		Vvec = V2[0:nb]

		return Vvec
	#------------------------------------------------------------
	def get_potential_from_grid(self, V_grid: np.ndarray) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		xp = self.xp
		nb = ne*ng-1
		w_i = self.leg.w_i
		x_i = self.leg.x_i

		Vvec = np.zeros(nb)
		V4 = np.zeros([ne, ng])

		# for i1 in range(ne-1):
		# 	w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i[ng]
		# 	w2 = 0.5*(xp[i1+2]-xp[i1+1]) * w_i[0]

		# 	V4[i1,0] = (V_grid[i1*ng + ng -1] * w1 + V_grid[(i1+1)*ng] * w2) / (w1 + w2)

		# for i1 in range(ne):
		# 	V4[i1,1:] = V_grid[i1*ng + 1 : i1*ng + ng]

		for i1 in range(1,ne):
			V4[i1-1,0] = V_grid[i1*ng]

		for i in range(ne):
			for m in range(1,ng):
				V4[i,m] = V_grid[i*ng+m]

		V2 = np.reshape(np.flip(V4, axis=1), [ne*ng])
		Vvec = V2[0:nb]

		return Vvec
	#------------------------------------------------------------
	def get_coeffs_from_func(self, f_fnc: callable) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		xp = self.xp
		nb = ne*ng-1
		w_i = self.leg.w_i
		x_i = self.leg.x_i

		f_vec = np.zeros(nb)
		f4 = np.zeros([ne,ng])

		for i1 in range(ne-1):
			w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i[ng]
			w2 = 0.5*(xp[i1+2]-xp[i1+1]) * w_i[0]
			xs1 = xp[i1] + 0.5 * (xp[i1+1] - xp[i1]) * (x_i[0:ng] + 1)
			xs2 = xp[i1+1] + 0.5 * (xp[i1+2] - xp[i1+1]) * (x_i[0:ng] + 1)

			f4[i1,0] = (f_fnc(xs1[-1])  + f_fnc(xs2[0]) ) / np.sqrt(w1 + w2)

		for i1 in range(ne):
			w = 0.5 * (xp[i1+1] - xp[i1]) * self.leg.w_i[:]
			xs = xp[i1] + 0.5 * (xp[i1+1] - xp[i1]) * (x_i[0:ng] + 1)
			f4[i1,1:] = f_fnc(xs[1:]) / np.sqrt(w[1:])

		f2 = np.reshape(np.flip(f4, axis=1), [ne*ng])
		f_vec = f2[0:nb]

		return f_vec
#------------------------------------------------------------
	def get_coeffs_from_func_batch(self, ndim: int, f_fnc: callable) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		xp = self.xp
		nb = ne*ng-1
		w_i = self.leg.w_i
		x_i = self.leg.x_i

		f4 = np.zeros([ndim, ne, ng])

		for i1 in range(ne-1):
			w1 = 0.5*(xp[i1+1]-xp[i1]) * w_i[ng]
			w2 = 0.5*(xp[i1+2]-xp[i1+1]) * w_i[0]
			xs1 = xp[i1] + 0.5 * (xp[i1+1] - xp[i1]) * (x_i[0:ng] + 1)

			f4[:, i1, 0] = f_fnc(xs1[-1]) *  np.sqrt(w1 + w2)

		for i1 in range(ne):
			w = 0.5 * (xp[i1+1] - xp[i1]) * self.leg.w_i[:]
			xs = xp[i1] + 0.5 * (xp[i1+1] - xp[i1]) * (x_i[0:ng] + 1)
			for m in range(1, ng):
				f4[:, i1, m] = f_fnc(xs[m]) * np.sqrt(w[m])

		f_vec = np.zeros([ndim, nb])
		for i in range(ndim):
			f2 = np.reshape(np.flip(f4[i, :, :], axis=1), [ne*ng])
			f_vec[i, :] = f2[0:nb]

		# f2 = np.reshape(np.flip(f4, axis=-1), [ndim, ne*ng])
		# f_vec = f2[0:ndim, 0:nb]

		return f_vec
	#------------------------------------------------------------
	def get_kinetic_energy_matrix(self, alpha: float = 0.0, beta: float = 0.0) -> np.ndarray:

		if alpha == 0.0 and  beta == 0.0:
			Tmat = self.__get_kinetic_energy_matrix_zerobound()
			return Tmat
		else:
			Tmat, Rvec = self.__get_kinetic_energy_matrix_asymbound(alpha, beta)
			return Tmat, Rvec
	#------------------------------------------------------------
	def get_kinetic_energy_banded(self) -> np.ndarray:
		Tmat = self.__get_kinetic_energy_matrix_zerobound()
		ne = self.ne
		ng = self.ng
		nb = ne * ng - 1

		# Determine the bandwidth
		upper_bw = ng

		Tmat_banded = np.zeros([upper_bw + 1, nb])

		# for j in range(nb):
		# 	for i in range(max(0, j - upper_bw), j + 1):
		# 		Tmat_banded[upper_bw + i - j, j] = Tmat[i, j]

		for k in range(upper_bw + 1):
			j_range = np.arange(k, nb)
			Tmat_banded[upper_bw - k, j_range] = np.diagonal(Tmat, offset=-k)

		return Tmat_banded
	#------------------------------------------------------------
	def get_deriv_matrix(self) -> np.ndarray:
		Dmat = self.__get_deriv_matrix_zerobound()
		return Dmat
	#------------------------------------------------------------
	def __get_kinetic_energy_matrix_zerobound(self):
		ne = self.ne
		ng = self.ng
		nb = ne*ng-1

		if not self.have_derivatives:
			self.der2_full = self.get_deriv_matrix_full(n=2, cplx=False)

		T2 = np.reshape(np.flip(self.der2_full, axis=(1,3)), [ne*ng,ne*ng])
		Tmat = T2[0:nb,0:nb]

		return Tmat
	#------------------------------------------------------------
	def __get_deriv_matrix_zerobound(self):
		ne = self.ne
		ng = self.ng
		nb = ne*ng-1

		if not self.have_derivatives:
			self.der1_full = self.get_deriv_matrix_full(n=1, cplx=False)

		T2 = np.reshape(np.flip(self.der1_full, axis=(1,3)), [ne*ng,ne*ng])
		# T2 = np.reshape(T4, [ne*ng,ne*ng])
		Dmat = T2[0:nb,0:nb]

		return Dmat
	#------------------------------------------------------------
	def __get_kinetic_energy_matrix_asymbound(self, alpha: float, beta: float) -> np.ndarray:
		ne = self.ne
		ng = self.ng
		xp = self.xp
		nb = ne*ng-1

		Le = xp[ne]-xp[ne-1]

		w1 = 0.5 * Le * self.leg.w_i[-1]
		w2 = 0.5 * Le * self.leg.w_i[-2]
		alg = np.sqrt(w1 / w2) * alpha
		btg = np.sqrt(w1) * beta

		if self.have_derivatives:
			T4 = np.array(self.der2_full, copy=True, dtype=np.complex128)
		else:
			T4 = self.get_deriv_matrix_full(n=2, cplx=True)

		tt = ttilde(self.leg,xp[ne]-xp[ne-1])

		w1 = 0.5*Le * self.leg.w_i
		w2 = 0.5*Le * self.leg.w_i[ng]
		Tne = 0.5 * tt[:,-1] / np.sqrt(w1[:] * w2)

		w1 = 0.5*(xp[ne-1]-xp[ne-2]) * self.leg.w_i[ng] + 0.5*(xp[ne]-xp[ne-1]) * self.leg.w_i[0]
		w2 = 0.5*(xp[ne]-xp[ne-1]) * self.leg.w_i[ng]
		Tne_mod = 0.5 * tt[0,-1] / np.sqrt(w1*w2)

		T4[ne-1,:,ne-1,ng-1] = T4[ne-1,:,ne-1,ng-1] + alg * Tne[0:ng]
		T4[ne-2,0,ne-1,ng-1] = T4[ne-2,0,ne-1,ng-1] + alg * Tne_mod

		T2 = np.reshape(np.flip(T4, axis=(1,3)), [ne*ng,ne*ng])
		Tmat = T2[0:nb,0:nb]

		R2 = np.zeros([ne,ng],dtype=np.complex128)
		R2[-1,:] = - btg * Tne[0:ng]
		R2[-2,0] = - btg * Tne_mod
		Rvec = np.reshape(np.flip( R2, axis=(1) ), [ne*ng] )[0:nb]

		return Tmat, Rvec
	#------------------------------------------------------------

#=================================================================
def lobatto_shape_derivatives(nodes):
    ng = len(nodes)
    df = np.zeros((ng, ng))  # df[m][n] = f'_m(x_n)
    for m in range(ng):
        for n in range(ng):
            if m != n:
                prod = 1.0
                for k in range(ng):
                    if k != m and k != n:
                        prod *= (nodes[n] - nodes[k]) / (nodes[m] - nodes[k])
                df[m, n] = prod / (nodes[m] - nodes[n])
            else:
                sum_term = 0.0
                for k in range(ng):
                    if k != m:
                        sum_term += 1.0 / (nodes[m] - nodes[k])
                df[m, n] = sum_term
    return df

#=================================================================
# Assemble first derivative matrix for one element
def local_first_derivative_matrix(nodes, weights):
    df = lobatto_shape_derivatives(nodes)
    ng = len(nodes)
    D = np.zeros((ng, ng))
    for m1 in range(ng):
        for m2 in range(ng):
            for j in range(ng):
                D[m1, m2] += df[m2, j] * weights[j] * (1.0 if j == m1 else 0.0)
    return D
#=================================================================
def dtilde(leg,L):
	D_ii = leg.D_ii
	w_i = leg.w_i
	x_i = leg.x_i

	# local_der1 = local_first_derivative_matrix(x_i, w_i)
	local_der1 = np.einsum('mn, m -> mn', D_ii, w_i)
	return local_der1

	# dt = np.einsum('nm, m -> mn', D_ii, w_i)
	# return 2. * dt
#=================================================================
def ttilde(leg,L):
	D_ii = leg.D_ii
	w_i = leg.w_i

	tt = np.einsum('ma,mb,m->ab',D_ii,D_ii,w_i)
	return 2 * tt/L
#=================================================================

