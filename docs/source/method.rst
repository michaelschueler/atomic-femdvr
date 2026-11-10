Method
======

Atomic units are used throughout (energies in Hartree, lengths in bohr).

Radial Kohn-Sham equations
--------------------------

For a spherical atom, the Kohn-Sham orbitals are
:math:`\psi_{nlm}(\mathbf r) = r^{-1} u_{nl}(r) Y_{lm}(\hat{\mathbf r})`, and the reduced
radial functions :math:`u_{nl}` solve

.. math::

    \left[-\frac{1}{2}\frac{d^2}{dr^2} + \frac{l(l+1)}{2r^2} + V_\mathrm{eff}(r)
    + \hat V^\mathrm{NL}_l\right] u_{nl}(r) = \varepsilon_{nl}\, u_{nl}(r),
    \qquad u_{nl}(0) = u_{nl}(R_\mathrm{max}) = 0,

with :math:`V_\mathrm{eff} = V_\mathrm{ion} + V_\mathrm{H} + V_\mathrm{xc}`. The density is
spherically averaged over the occupied shells with occupations :math:`f_{nl}`,
:math:`\rho(r) = \sum_{nl} f_{nl}\, u_{nl}^2(r)/r^2`.

- **All-electron**: :math:`V_\mathrm{ion} = -Z/r` and :math:`\hat V^\mathrm{NL}_l = 0`.
- **Pseudo-atom**: :math:`V_\mathrm{ion}` is the local part of the pseudopotential and
  :math:`\hat V^\mathrm{NL}_l = \sum_{ij} |\beta_i\rangle D_{ij} \langle\beta_j|` is the
  separable non-local part, summed over the projectors of angular momentum :math:`l` in the
  UPF file.

Hartree and exchange-correlation potential
------------------------------------------

The Hartree potential follows from radial integration,

.. math::

    V_\mathrm{H}(r) = \frac{1}{r}\int_0^r \rho(r')\, r'^2 dr' + \int_r^{R_\mathrm{max}} \rho(r')\, r'\, dr'.

The exchange-correlation potential is evaluated for the density plus, if present in the
UPF file, the partial core density (non-linear core correction). For a GGA with energy
density :math:`e_\mathrm{xc}(n, \sigma)`, :math:`\sigma = |\nabla n|^2`, the potential
includes the gradient term

.. math::

    V_\mathrm{xc}(r) = \frac{\partial e_\mathrm{xc}}{\partial n}
    - \frac{2}{r^2}\frac{d}{dr}\left(r^2\, \frac{\partial e_\mathrm{xc}}{\partial \sigma}\,
    \frac{dn}{dr}\right).

PBE [PBE96]_ is implemented internally; other LDA and GGA functionals are taken from libxc.

Finite-element discrete variable representation
-----------------------------------------------

The interval :math:`[0, R_\mathrm{max}]` is divided into finite elements. In each element
the functions are expanded in Lagrange polynomials on ``ng + 1`` Gauss-Lobatto points;
neighbouring elements share their boundary point (bridge functions), which makes the
basis continuous [RM00]_. With Gauss-Lobatto quadrature, local potentials are diagonal and
the kinetic energy couples only points of the same element, so the Hamiltonian is banded.
Eigenvalues converge rapidly with ``ng``; ``ng = 12`` gives about 1e-7 Ha.

The element sizes adapt to the atom: they grow from ``h_min`` near the nucleus to ``h_max``
far from it, controlled by a weight function :math:`e^{-Z r} + e^{-r}` (:math:`Z` is the
valence charge for pseudo-atoms) that is integrated with an adaptive step-size control
of tolerance ``elem_tol``. All-electron grids start with a uniform core region up to
:math:`r = 5/Z`; their size parameters are scaled by :math:`Z^{-1/3}`.

Self-consistency
----------------

Starting from the atomic density of the UPF file (pseudo-atoms) or hydrogen-like shells
with Slater-screened charges (all-electron), the Kohn-Sham equations are iterated until the
density residual :math:`\|\rho_\mathrm{out} - \rho_\mathrm{in}\|_2` drops below
``conv_tol``. The new input density is obtained by linear or Anderson (Pulay) mixing
[And65]_.

Pseudopotentials without pseudo-atomic orbitals (e.g. SG15 [SG15]_) do not specify
occupations. For these, the atomic density stored in the UPF file (``PP_RHOATOM``) is used
without self-consistency, and the valence channels are taken from the angular momenta of
the projectors.

Confined orbitals and polarization functions
--------------------------------------------

For use as localized basis functions, the orbitals of the ``nscf`` task are computed with
an additional confinement potential that vanishes for :math:`r < r_i` and rises up to the
confinement radius :math:`r_c`:

.. math::

    V_\mathrm{conf}(r) =
    \begin{cases}
        V_b \sin^2\!\left(\dfrac{\pi}{2}\dfrac{r - r_i}{r_c - r_i}\right) & \text{softstep}\\[1ex]
        \left(\dfrac{r - r_i}{r_c - r_i}\right)^2 & \text{harmonic}
    \end{cases}
    \qquad r_i \le r < r_c,

and is constant (:math:`V_b`, or 1000 Ha for the harmonic potential) beyond :math:`r_c`.
The resulting energy shift :math:`\varepsilon^\mathrm{conf} - \varepsilon^\mathrm{free}` of
the highest occupied state is printed for each valence channel.

Unoccupied channels above the valence (polarization channels) are usually unbound or only
weakly bound in the atomic potential. With ``polarization_mode = "softcoul"``, their radial
functions are computed in the soft-Coulomb potential

.. math::

    V_\mathrm{sc}(r) = -\frac{Q}{\sqrt{r^2 + \delta^2}}

(plus confinement), which yields compact orbitals. The reported eigenvalues of these
channels are those of the confined Kohn-Sham Hamiltonian; they serve as an energy
criterion for selecting orbitals for a low-energy subspace. The ``optimize`` task chooses
:math:`Q` such that the lowest soft-Coulomb orbital in the highest valence channel has
maximal overlap with the corresponding Kohn-Sham orbital.

Relativistic treatments (all-electron, approximate)
---------------------------------------------------

Two scalar-relativistic approximations without spin-orbit coupling are available. They
have not been benchmarked against reference data.

- **ZORA** [ZORA93]_: the kinetic energy becomes
  :math:`-\tfrac{1}{2}\tfrac{d}{dr} M(r) \tfrac{d}{dr}` with
  :math:`M(r) = [1 - V(r)/2c^2]^{-1}`; the centrifugal term is scaled by the same factor.
- **Scalar-relativistic** (Koelling-Harmon type [KH77]_): the mass factor
  :math:`[1 + (\varepsilon - V(r))/2c^2]^{-1}` depends on the energy. It is evaluated at a
  common reference energy per channel and iterated to a fixed point, which keeps the
  orbitals of a channel orthogonal.

In both cases the nuclear potential in the mass factor is replaced by that of a Gaussian
nuclear charge of width ``nuclear_sigma``, which keeps the mass factor finite at the
nucleus.

Photoemission
-------------

Continuum states at kinetic energy :math:`E = k^2/2` are computed in the same potential
(including the non-local pseudopotential) with outgoing-wave boundary conditions at
:math:`R_\mathrm{max}`, expressed by spherical Hankel functions. The scattering phase
shifts :math:`\delta_l` follow from :math:`S_l = e^{2i\delta_l}`. Radial dipole matrix
elements between occupied orbitals and continuum states with :math:`l_f = l_i \pm 1` are
evaluated in the length gauge, :math:`\int u_f^* \, r\, u_i\, dr`, and/or the velocity
gauge.

References
----------

.. [RM00] T. N. Rescigno and C. W. McCurdy, Phys. Rev. A **62**, 032706 (2000).
.. [PBE96] J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. **77**, 3865 (1996).
.. [And65] D. G. Anderson, J. ACM **12**, 547 (1965).
.. [SG15] M. Schlipf and F. Gygi, Comput. Phys. Commun. **196**, 36 (2015).
.. [ZORA93] E. van Lenthe, E. J. Baerends, and J. G. Snijders, J. Chem. Phys. **99**, 4597 (1993).
.. [KH77] D. D. Koelling and B. N. Harmon, J. Phys. C **10**, 3107 (1977).
