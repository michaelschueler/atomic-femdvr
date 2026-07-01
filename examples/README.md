# Examples

Each `*.json` file in this directory is a complete input that can be passed
directly to the `atomic_femdvr` console script. Run from the repository
root:

```console
$ atomic_femdvr atomic        examples/Ne_full.json -t scf
$ atomic_femdvr atomic        examples/C_full.json  -t scf
$ atomic_femdvr pseudoatomic  examples/S_Pseudo.json -t scf -t nscf
```

| File              | Mode           | Element | XC          | Notes                                                                 |
|-------------------|----------------|---------|-------------|-----------------------------------------------------------------------|
| `Ne_full.json`    | `atomic`       | Ne      | LDA (PZ)    | Closed-shell; Anderson mixing.                                        |
| `C_full.json`     | `atomic`       | C       | LDA (PZ)    | Open-shell; useful for SCF debugging.                                 |
| `S_Pseudo.json`   | `pseudoatomic` | S       | GGA (PBE)   | Reads `data/S.upf` (PseudoDojo NC-SR 0.4 PBE standard, ONCVPSP, GPL). |

## Bundled pseudopotentials

`data/S.upf` is from the [PseudoDojo](http://www.pseudo-dojo.org)
NC-SR 0.4 PBE *standard* set, generated with ONCVPSP and licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
