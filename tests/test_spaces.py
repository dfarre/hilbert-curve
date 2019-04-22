import unittest

import pandas

from hilbert import spaces


class TestReals(unittest.TestCase):
    def test_from_index(self):
        index = pandas.Index([1, 3, 4, 6, 9])
        domain = spaces.Reals.from_index(index)

        assert (domain.support == index.array).all()
        assert domain.measure == (2 + 1 + 2 + 3)/4

    def test_from_range_index(self):
        index = pandas.RangeIndex(3, 11)
        domain = spaces.Reals.from_index(index)

        assert (domain.support == index.array).all()
        assert domain.measure == index._step
