import pandas

from hilbert import algebra
from hilbert import stock

from hilbert.curves import lib

EXPONENTS = {0: '', 1: '', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹'}


@stock.FrozenLazyAttrs(('curves',))
class CurveVector(algebra.Vector):
    def __init__(self, space, *curves, image=None):
        super().__init__()
        self.space = space

        if isinstance(image, algebra.Image):
            stock.FrozenLazyAttrs.set('image', self, image)
            self.curves = ()
        else:
            self.curves = curves

    @stock.Attr.getter
    def real(self, i):
        return pandas.Series(numpy.real(i.to_numpy()), index=i.index)

    @stock.Attr.getter
    def imag(self, i):
        return pandas.Series(numpy.imag(i.to_numpy()), index=i.index)

    @stock.Attr.getter
    def density(self, i):
        return i.abs()**2


class RImage(Image):
    def __getitem__(self, x):
        return self.i.loc[x]


class CImage(Image):
    def __getitem__(self, z):
        return self.i.loc[(z.real, z.imag)]


@stock.FrozenAttrs('space', 'image_type')
class Vector(stock.Repr, stock.Attr, stock.Eq, algebra.Vector):
    def __init__(self, space, *curves):
        super().__init__()
        self.space, self.curves = space, curves
        self.image_type = CImage if isinstance(self.space.bases, spaces.C1Field) else RImage

    def __call__(self, x):
        return self.image[x]

    def __str__(self):
        return ' + '.join(list(map(str, self.curves)))

    @stock.Attr.getter
    def image(self, curves):
        return self.image_type(series=sum(map(self.get_image_series, curves)))

    @stock.Attr.getter
    def norm(self, image):
        return numpy.sqrt(self@self)

    def get_image(self, curve):
        return algebra.Image(curve(self.space.domain), index=self.space.index)

    def update(self):
        """Called to propagate space mutations when needed"""
        if self._frozen['image']:
            self.image.i = self.image.i.reindex(index=self.space.index)
            nans = self.image.i[self.image.i.isna()]
            values = pandas.Series(sum([
                c(self.space.bases.index_domain(nans.index)) for c in self.curves
            ]) if self.curves else [0]*len(nans.index), index=nans.index)
            self.image.i.update(values)

        return self

    def kind(self):
        return '+'.join(sorted(curve.kind() for curve in self.curves))

    def eat(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.space, lib.Polynomial(other))

        if isinstance(other, algebra.Vector):
            return other.stack(self.space)

        return super().eat(other)

    def num_prod(self, number):
        if self.curves:
            return self.__class__(self.space, *(cu.num_prod(number) for cu in self.curves))

        return self.__class__(self.space, image=number*self.image)

    def add_other(self, other):
        if self.curves and other.curves:
            return self.__class__(self.space, *(self.curves + other.curves))

        return self.__class__(self.space, image=self.image+other.image)
