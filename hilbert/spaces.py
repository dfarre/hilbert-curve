import abc

import numpy

from hilbert.curves import base


class Reals:
    def __init__(self, start, end, cell):
        self.support = numpy.arange(start, end, cell)
        self.measure = cell


class ComplexRectangle:
    def __init__(self, sw, ne, re_cell, im_cell=None):
        self.re_cell, self.im_cell = re_cell, im_cell or re_cell
        self.size = tuple(map(int, ((ne.real - sw.real)/self.re_cell,
                                    (ne.imag - sw.imag)/self.im_cell)))
        self.support = numpy.array([[
            (sw.real + x*self.re_cell) + (sw.imag + y*self.im_cell)*1j
            for x in range(self.size[0])] for y in range(self.size[1])])
        self.measure = self.re_cell*self.im_cell


class NoFiniteVectorNorm(Exception):
    """Raised for wrong vector input"""


class Space(metaclass=abc.ABCMeta):
    def __init__(self, validate_norm=True):
        self.validate_norm = validate_norm

    def __call__(self, *args, **kwargs):
        vector = self.make_vector(*args, **kwargs)

        if self.validate_norm:
            self.validate(vector)

        return vector

    @abc.abstractmethod
    def make_vector(self, *args, **kwargs):
        """Vector instance constructor returning the new vector"""

    def validate(self, vector, validate_norm=True):
        braket = vector @ vector

        if not (braket > 0 and numpy.isfinite(braket)):
            raise NoFiniteVectorNorm(
                f'{vector} does not belong to the space - no finite norm!')


class LebesgueCurveSpace(Space):
    def __init__(self, curve_domain, validate_norm=True):
        super().__init__(validate_norm)
        self.domain = curve_domain

    def make_vector(self, *args, **kwargs):
        return base.Vector(self.domain, *args, **kwargs)
