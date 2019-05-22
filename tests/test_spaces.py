import unittest

import pandas

from ipynb_tests import tester

from hilbert import spaces


class BasisChangeTests(tester.NotebookTester, unittest.TestCase):
    notebooks_path = 'notebooks'


class TestR1Field(unittest.TestCase):
    def test_from_index(self):
        index = pandas.Index([1, 3, 4, 6, 9])
        space = spaces.R1Field.from_index(spaces.LebesgueCurveSpace, index)

        assert (space.bases.o.index == index).all()
        assert space.bases.measure == (2 + 1 + 2 + 3)/4

    def test_from_range_index(self):
        index = pandas.RangeIndex(3, 11)
        space = spaces.R1Field.from_index(spaces.LebesgueCurveSpace, index)

        assert (space.bases.o.index == index).all()
        assert space.bases.measure == index._step


class SpaceGetitemTests(unittest.TestCase):
    error = r"Requested new vector\(s\) for non-analytic basis 'foo'"

    def test_key_error__value(self):
        C2 = spaces.R1Field.range(spaces.LebesgueCurveSpace, 0, 1, 2)

        self.assertRaisesRegex(KeyError, self.error, C2.__getitem__, (0, 'foo'))
        assert C2.bases.o['foo'].isna().all()  # column added anyway
        self.assertRaisesRegex(KeyError, self.error, C2.__getitem__, (0, 'foo'))

    def test_key_error__slice(self):
        C2 = spaces.R1Field.range(spaces.LebesgueCurveSpace, 0, 1, 2)

        self.assertRaisesRegex(KeyError, self.error, C2.__getitem__, (slice(0, 1), 'foo'))
        C2.bases[0:1, 'foo'] = ':)'
        assert super(spaces.R1Field, C2.bases).__str__() == '\n    foo\n0.0  :)\n1.0  :)'
        self.assertRaisesRegex(KeyError, self.error, C2.__getitem__, (slice(0, 1), 'foo'))


class SpaceScalingTests(unittest.TestCase):
    def test_real(self):
        R0to1L2 = spaces.R1Field.range(spaces.LebesgueCurveSpace, 0, 1, 101)
        u, v, w = R0to1L2[0.87, 'delta'], R0to1L2[0.0, 'delta'], R0to1L2[0.13, 'delta']

        assert (u@u, v@v, w@w) == (1, 1, 1)

        R0to1L2.scale(0.56)

        assert (u@u, v@v, w@w) == (0.56, 0.56, 0.56)

        a, b, c = (R0to1L2[0.56*0.87, 'delta'],
                   R0to1L2[0.0, 'delta'],
                   R0to1L2[0.56*0.13, 'delta'])

        assert tuple(map(lambda x: round(x, 7), (a@a, b@b, c@c))) == (1, 1, 1)

    def test_complex(self):
        C1L2 = spaces.C1Field.rectangle(spaces.LebesgueCurveSpace, -1 - 1j, 1 + 1j, 201)
        u, v, w = C1L2[0.49-0.38j, 'delta'], C1L2[0.0, 'delta'], C1L2[-0.83j, 'delta']

        assert (u@u, v@v, w@w) == (1, 1, 1)

        C1L2.scale(2.3)

        assert (u@u, v@v, w@w) == (2.3**2, 2.3**2, 2.3**2)

        u, v, w = (C1L2[2.3*0.49-2.3*0.38j, 'delta'],
                   C1L2[0.0, 'delta'],
                   C1L2[-2.3*0.83j, 'delta'])

        assert tuple(map(lambda x: round(x, 7), (u@u, v@v, w@w))) == (1, 1, 1)
