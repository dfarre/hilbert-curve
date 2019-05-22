import abc

import pandas
import numpy

from hilbert import EQ_ROUND_TO

from hilbert import algebra
from hilbert import spaces
from hilbert import stock

from hilbert.curves import lib

EXPONENTS = {0: '', 1: '', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹'}


class Image(stock.Repr, metaclass=abc.ABCMeta):
    def __init__(self, *args, series=None, **kwargs):
        self.i = pandas.Series(*args, **kwargs) if series is None else series

    def __str__(self):
        return f'\n{self.i}'

    @abc.abstractmethod
    def __getitem__(self, x):
        """Get image value at `x`"""

    @property
    def real(self):
        return pandas.Series(self.i.real, index=self.i.index)

    @property
    def imag(self):
        return pandas.Series(self.i.imag, index=self.i.index)


class RImage(Image):
    def __getitem__(self, x):
        return self.i.loc[x]


class CImage(Image):
    def __getitem__(self, z):
        return self.i.loc[(z.real, z.imag)]


@stock.FrozenLazyAttrs(('space', 'curves', 'image_type'), ('image',))
class Vector(stock.Repr, stock.Eq, algebra.Vector):
    def __init__(self, space, *curves):
        self.space, self.curves = space, curves
        self.image_type = CImage if isinstance(self.space.bases, spaces.C1Field) else RImage

    def __call__(self, x):
        return self.image[x]

    def __str__(self):
        return ' + '.join(list(map(str, self.curves)))

    def _make_image(self):
        return self.image_type(series=sum(map(self.get_image_series, self.curves)))

    def get_image_series(self, curve):
        if isinstance(curve, lib.ImageCurve):
            return curve.image.i

        return self.image_type(
            curve(self.space.bases.domain()), index=self.space.bases.o.index).i

    def eq(self, other):
        return self.space == other.space and not (
            self.image.i - other.image.i).abs().round(EQ_ROUND_TO).any()

    def update(self):
        """Called to propagate space mutations when needed"""
        image = list(filter(lambda c: isinstance(c, lib.ImageCurve), self.curves))

        for curve in image:
            curve.image.i = curve.image.i.reindex(
                index=self.space.bases.o.index, fill_value=0)

        if self._frozen['image']:
            self.image.i = self.image.i.reindex(index=self.space.bases.o.index)
            nans = self.image.i[self.image.i.isna()]
            analytic = list(filter(
                lambda c: not isinstance(c, lib.ImageCurve), self.curves))
            values = pandas.Series(sum([
                c(self.space.bases.index_domain(nans.index)) for c in analytic
            ]) if analytic else [0]*len(nans.index), index=nans.index)
            self.image.i.update(values)

        return self

    def full_plot(self, top_ax, bottom_ax, **kwargs):
        self.image.real.plot(ax=top_ax, **kwargs)
        self.image.imag.plot(ax=bottom_ax, **kwargs)

    @stock.PyplotShow(nrows=2, ncols=1)
    def plot(self, **kwargs):
        top_ax, bottom_ax = kwargs.pop('axes')
        self.full_plot(top_ax, bottom_ax, **kwargs)
        top_ax.yaxis.set_label_text('Re')
        bottom_ax.yaxis.set_label_text('Im')
        top_ax.grid()
        bottom_ax.grid()

    def kind(self):
        return '+'.join(sorted(curve.kind() for curve in self.curves))

    def eat(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.space, lib.Polynomial(other))

        return other

    def num_prod(self, number):
        return self.__class__(self.space, *(number*cu for cu in self.curves))

    def add_other(self, other):
        self.check_if_equal_spaces(other)

        return self.__class__(self.space, *(self.curves + other.curves))

    def braket(self, other):
        self.check_if_equal_spaces(other)

        return self.space.bases.measure*numpy.dot(numpy.conj(self.image.i), other.image.i)

    def check_if_equal_spaces(self, other):
        if not self.space == other.space:
            raise NotImplementedError('Vectors should belong to the same space')
