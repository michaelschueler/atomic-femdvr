# Test pseudopotentials

The norm-conserving pseudopotentials in this directory are taken unmodified from the
[PseudoDojo](http://www.pseudo-dojo.org) table (NC ONCVPSP v0.4, scalar-relativistic, PBE):

- `Mo/Mo.upf`
- `Si/Si.oncvpsp.upf`

They are distributed under the Creative Commons Attribution 4.0 International license
(CC BY 4.0), not under the MIT license of the code. If you use them, please cite:

- M. J. van Setten, M. Giantomassi, E. Bousquet, M. J. Verstraete, D. R. Hamann, X. Gonze,
  and G.-M. Rignanese, Comput. Phys. Commun. **226**, 39 (2018),
  https://doi.org/10.1016/j.cpc.2018.01.012
- D. R. Hamann, Phys. Rev. B **88**, 085117 (2013) (ONCVPSP)

## Regularized all-electron potential

`Si/Si_ANC_a4.upf` is Gygi's analytic norm-conserving (ANC) regularized nuclear potential
for Si with a = 4. It is local-only (`number_of_proj="0"`, no `PP_NONLOCAL`) and keeps the
full nuclear charge Z = 14. It replaces -Z/r by a smooth potential that is finite at
the nucleus. It was generated for this project from the analytic formula in

- F. Gygi, J. Chem. Theory Comput. **19**, 1300 (2023),
  https://doi.org/10.1021/acs.jctc.2c01191
- S. Lehtola, J. Chem. Theory Comput. **19**, 4033 (2023),
  https://doi.org/10.1021/acs.jctc.3c00530 (converged b(a) and accuracy benchmarks)
