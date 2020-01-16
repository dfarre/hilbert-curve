import unittest

from hilbert import fields
from hilbert import spaces


class SpaceGetitemTests(unittest.TestCase):
    error = r"Requested new vector\(s\) for non-analytic basis 'foo'"

    def test_key_error__value(self):
        C2 = spaces.CnSpace(2)

        self.assertRaisesRegex(KeyError, self.error, C2.__getitem__, (1, 'foo'))
        assert C2.bases.o['foo'].isna().all()  # column added anyway
        self.assertRaisesRegex(KeyError, self.error, C2.__getitem__, (1, 'foo'))

    def test_key_error__slice(self):
        C2 = spaces.CnSpace(2)

        self.assertRaisesRegex(KeyError, self.error, C2.__getitem__, (slice(1, 2), 'foo'))
        C2.bases[1:2, 'foo'] = ':)'
        assert super(fields.R1Field, C2.bases).__str__() == '\n    foo\n1.0  :)\n2.0  :)'
        self.assertRaisesRegex(KeyError, self.error, C2.__getitem__, (slice(1, 2), 'foo'))


class SpaceScalingTests(unittest.TestCase):
    def test_real(self):
        R0to1L2 = spaces.R1LebesgueSpace(0, 1, 101)
        u, v, w = R0to1L2[0.87, 'delta'], R0to1L2[0.0, 'delta'], R0to1L2[0.13, 'delta']

        assert (u@u, v@v, w@w) == (1, 1, 1)

        R0to1L2.scale(0.56)

        assert (u@u, v@v, w@w) == (0.56, 0.56, 0.56)

        a, b, c = (R0to1L2[0.56*0.87, 'delta'],
                   R0to1L2[0.0, 'delta'],
                   R0to1L2[0.56*0.13, 'delta'])

        assert tuple(map(lambda x: round(x, 7), (a@a, b@b, c@c))) == (1, 1, 1)

    def test_complex(self):
        C1L2 = spaces.R2LebesgueSpace(-1 - 1j, 1 + 1j, 201)
        u, v, w = C1L2[0.49-0.38j, 'delta'], C1L2[0.0, 'delta'], C1L2[-0.83j, 'delta']

        assert (u@u, v@v, w@w) == (1, 1, 1)

        C1L2.scale(2.3)

        assert (u@u, v@v, w@w) == (2.3**2, 2.3**2, 2.3**2)

        u, v, w = (C1L2[2.3*0.49-2.3*0.38j, 'delta'],
                   C1L2[0.0, 'delta'],
                   C1L2[-2.3*0.83j, 'delta'])

        assert tuple(map(lambda x: round(x, 7), (u@u, v@v, w@w))) == (1, 1, 1)
