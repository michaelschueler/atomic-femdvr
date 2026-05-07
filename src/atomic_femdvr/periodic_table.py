"""Minimal element-symbol to atomic-number lookup (first 20 elements)."""


# ==========================================================================
class PeriodicTable:
    """Lookup table mapping chemical symbols to atomic numbers."""

    # .......................................................
    @staticmethod
    def get_atomic_number(element: str) -> int:
        """Return the atomic number ``Z`` for the given element symbol."""
        periodic_table = {
            "H": 1,
            "He": 2,
            "Li": 3,
            "Be": 4,
            "B": 5,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Ne": 10,
            "Na": 11,
            "Mg": 12,
            "Al": 13,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Ar": 18,
            "K": 19,
            "Ca": 20,
            # ... (extend as needed)
        }
        if element not in periodic_table:
            raise ValueError(f"Element '{element}' not found in periodic table.")
        return periodic_table[element]

    # .......................................................
