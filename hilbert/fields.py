import abc
import itertools

import pandas
import numpy

from hilbert import INDEX_ROUND_TO

from hilbert import algebra
from hilbert import stock


class Field(stock.IndexXFrame, metaclass=abc.ABCMeta):
    dtype = None

    @property
    def dimension(self):
        return len(self.o.index)

    @abc.abstractproperty
    def measure(self):
        """The cell measure"""

    @abc.abstractproperty
    def bounds(self):
        """Return the boundary values as [min, max], ..."""

    @abc.abstractproperty
    def limits(self):
        """Return the 'open' limits as (l, r), ..."""

    @abc.abstractproperty
    def cell_limits(self):
        """Return cell boundary values as (b0, b1, ...), ... for each dimension"""

    @abc.abstractmethod
    def scale_index(self, k):
        """Scale the index by `k`"""

    @abc.abstractmethod
    def scale_cell(self, k):
        """Scale the unit cell length by `k`"""

    @abc.abstractmethod
    def replicas(self, copies):
        """Replicate the index from its boundaries - return array of replicas"""

    @abc.abstractmethod
    def yield_vector_replicas(self, key, copies, index_replicas):
        """Yield all new `z, series` tuples for basis `key`"""

    @abc.abstractmethod
    def extend_index(self, copies, index_replicas):
        """Reindex from the replicas"""

    @abc.abstractmethod
    def plot_domain(self, **kwargs):
        """Show space domain"""

    def update(self):
        self.o = self.o.apply(lambda s: s.map(
            lambda v: v.update() if isinstance(v, algebra.Vector) else v))


class R1Field(stock.RealIndexMixin, Field):
    dtype = numpy.float64

    def __init__(self, cell, *args, **kwargs):
        self.cell = cell
        super().__init__(pandas.DataFrame(*args, **kwargs))

    def __str__(self):
        return (f'ℝ-segment | cell {round(self.cell, 9)} with bounds {self.bounds}'
                f' | D = {self.dimension}')

    @property
    def measure(self):
        return self.cell

    @property
    def limits(self):
        mi, ma = self.bounds

        return (mi - self.cell/2, ma + self.cell/2)

    @property
    def cell_limits(self):
        return tuple(self.o.index - self.cell/2) + self.limits[1:]

    @property
    def bounds(self):
        return [self.o.index.min(), self.o.index.max()]

    def scale_index(self, k):
        self.o.index *= k

    def scale_cell(self, k):
        self.cell *= k

    def replicas(self, copies):
        l, r = self.bounds

        return [numpy.around(self.o.index + n*(r - l + self.measure), INDEX_ROUND_TO)
                for n in range(-copies, copies + 1)]

    def yield_vector_replicas(self, key, copies, index_replicas):
        current_vectors = self.o[key][~self.o[key].isna()]
        cases = itertools.product(enumerate(current_vectors), index_replicas)

        for (k, vec), index in cases:
            series = pandas.Series(vec.image.i.to_numpy(), index=index)

            yield index[k], series

    def extend_index(self, copies, index_replicas):
        self.o = self.o.reindex(
            index=self.o.index.__class__(itertools.chain(*index_replicas)))

    @stock.PyplotShow(figsize=(25, 1))
    def plot_domain(self, **kwargs):
        axes = self.base_plot(kwargs.pop('axes'))
        axes.grid(which='minor', color='w')
        axes.patch.set_facecolor('green')

    def base_plot(self, axes, title=None):
        axes.set_title(title or str(self))
        axes.xaxis.set_ticks(self.cell_limits, minor=True)
        axes.yaxis.set_ticks([])
        axes.set_xlim(*self.limits)
        axes.set_ylim(-self.cell/2, self.cell/2)
        axes.set_aspect('equal')
        axes.xaxis.set_label_text('Re')

        return axes

    @classmethod
    def range(cls, space_type, start, end, cells):
        arr, cell = numpy.linspace(start, end, cells, retstep=True)

        return space_type(cls(cell, index=pandas.Index(
            numpy.around(arr, INDEX_ROUND_TO)).astype(cls.dtype)))

    @classmethod
    def from_index(cls, space_type, index):
        return space_type(cls(
            index._step if isinstance(index, pandas.RangeIndex)
            else abs((index[1:] - index[:-1]).array).mean(), index=index))


class C1Field(stock.ComplexIndexMixin, Field):
    dtype = numpy.complex128

    def __init__(self, re_cell, im_cell, *args, **kwargs):
        self.re_cell, self.im_cell = re_cell, im_cell
        super().__init__(pandas.DataFrame(*args, **kwargs))

    def __str__(self):
        return (f'ℂ-rectangle | cell {self.re_cell}x{self.im_cell} with '
                f'bounds {self.bounds} | D = {self.dimension}')

    @property
    def measure(self):
        return self.re_cell*self.im_cell

    @property
    def limits(self):
        (rmi, rma), (imi, ima) = self.bounds

        return ((rmi - self.re_cell/2, rma + self.re_cell/2),
                (imi - self.im_cell/2, ima + self.im_cell/2))

    @property
    def cell_limits(self):
        (rl, rr), (il, ir) = self.limits

        return (tuple(self.o.index.levels[0] - self.re_cell/2) + (rr,),
                tuple(self.o.index.levels[1] - self.im_cell/2) + (ir,))

    @property
    def bounds(self):
        return [[self.o.index.levels[0].min(), self.o.index.levels[0].max()],
                [self.o.index.levels[1].min(), self.o.index.levels[1].max()]]

    def scale_cell(self, k):
        self.re_cell *= k
        self.im_cell *= k

    def scale_index(self, k):
        self.o.index = self.o.index.map(lambda lab: (k*lab[0], k*lab[1]))

    def extend_index(self, copies, index_replicas):
        self.o = self.o.reindex(index=pandas.MultiIndex.from_product(
            [itertools.chain(*index_replicas[0]), itertools.chain(*index_replicas[1])],
            names=self.o.index.names))

    def replicas(self, copies):
        (rmi, rma), (imi, ima) = self.bounds
        rng = range(-copies, copies + 1)
        real = [numpy.around(
            self.o.index.levels[0] + m*(rma - rmi + self.re_cell), INDEX_ROUND_TO
        ) for m in rng]
        imag = [numpy.around(
            self.o.index.levels[1] + n*(ima - imi + self.im_cell), INDEX_ROUND_TO
        ) for n in rng]

        return real, imag

    def yield_vector_replicas(self, key, copies, index_replicas):
        current_vectors = self.o[key][~self.o[key].isna()]
        cases = itertools.product(enumerate(current_vectors), *index_replicas)

        for (k, vec), xrepl, yrepl in cases:
            index = pandas.MultiIndex.from_product([xrepl, yrepl])
            series = pandas.Series(vec.image.i.to_numpy(), index=index)

            yield complex(*index[k]), series

    def base_plot(self, axes, title=None):
        axes.set_title(title or str(self))
        xticks, yticks = self.cell_limits
        axes.xaxis.set_ticks(xticks, minor=True)
        axes.yaxis.set_ticks(yticks, minor=True)
        (rl, rr), (il, ir) = self.limits
        axes.set_xlim(rl, rr)
        axes.set_ylim(il, ir)
        axes.set_aspect('equal')
        axes.xaxis.set_label_text('Re')
        axes.yaxis.set_label_text('Im')

        return axes

    @stock.PyplotShow(figsize=(8, 8))
    def plot_domain(self, **kwargs):
        axes = self.base_plot(kwargs.pop('axes'))
        axes.grid(which='minor', color='w')
        axes.patch.set_facecolor('green')

    @stock.PyplotShow(figsize=(8, 8))
    def density_plot(self, vector, **kwargs):
        title = repr(vector)
        axes = self.base_plot(
            kwargs.pop('axes'), (title[:120] + '...') if len(title) > 120 else title)
        values = vector.image.density.unstack(level=0)
        xticks, yticks = self.cell_limits
        coll = axes.pcolormesh(xticks, yticks, values.values, **kwargs)
        kwargs.pop('figure').colorbar(coll, ax=axes)

    @classmethod
    def rectangle(cls, space_type, sw, ne, re_cells, im_cells=None):
        reals, re_cell = numpy.linspace(sw.real, ne.real, re_cells, retstep=True)
        imags, im_cell = numpy.linspace(sw.imag, ne.imag, im_cells or re_cells, retstep=True)
        index = pandas.MultiIndex.from_product([
            numpy.around(reals, INDEX_ROUND_TO), numpy.around(imags, INDEX_ROUND_TO)
        ], names=('real', 'imag'))

        return space_type(cls(re_cell, im_cell, index=index))
