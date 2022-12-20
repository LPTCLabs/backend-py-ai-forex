import unittest

from src import util


class UtilsTest(unittest.TestCase):
    def test_utils(self) -> None:
        repeated_values: list = [1, 1, 2, 2]
        self.assertEqual(
            util.keep_unrepeated_values(repeated_values),
            [1, 2],
            "Error in deleting repeated values.",
        )
