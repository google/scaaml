"""Counts how many times each value of each attack point appears."""


class APChecker:
    """Checks that the attack point is random enough based on counts.

    Args:
      counts: Numpy array of counts of shape (len, max_val) where
        counts[i][val] counts how many times does the i-th piece (e.g., byte
        or bit) attain value val.
      attack_point_name: The name of the attack point. Useful for debugging
        purposes.

    Example use:
    km_checker = APChecker(counts=km_counter.get_counts(),
                           attack_point_name='km')
    """
    def __init__(self, counts, attack_point_name: str) -> None:
        self._counts = counts.copy()
        self._attack_point_name = attack_point_name
        self.run_all()

    def run_all(self):
        """Run all statistical checks. When adding a new test remember to call
        it from this method. To test that your method is called from run_all,
        take a look at
        tests/stats/test_ap_checker.py::test_run_all_calls_check_all_nonzero.

        Raises: If any check raises.
        """
        self.check_all_nonzero()

    @property
    def attack_point_name(self) -> str:
        """Returns the name of the attack point."""
        return self._attack_point_name

    def check_all_nonzero(self):
        """Check that every value of the attack point appears at least once.

        Raises: ValueError if there is a value of an attack point that is
          never present.
        """
        # The ap_piece is either byte or bit of an attack point (e.g. single
        # byte of the key).
        for ap_piece_number, ap_piece in enumerate(self._counts):
            if not (ap_piece > 0).all():
                msg = (
                    f'Not all combinations of attack_point-value appear, for '
                    f'example {self._attack_point_name}_{ap_piece_number} '
                    f'never has value {(ap_piece > 0).argmin()}.')
                raise ValueError(msg)
