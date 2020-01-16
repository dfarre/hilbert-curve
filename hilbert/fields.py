import abc
import itertools

import pandas
import numpy

from hilbert import INDEX_ROUND_TO

from hilbert import algebra
from hilbert import stock


class Field(stock.WrappedDataFrame, metaclass=abc.ABCMeta):
    dtype = None

    @property
    def dimension(self):
        return len(self.o.index)

    @abc.abstractmethod
    def plot_domain(self, **kwargs):
        """Show space domain"""

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
    def replicate_index(self, copies):
        """Create and return an extended index by replication"""

    def extend_index(self, copies):
        self.o = self.o.reindex(index=self.replicate_index(copies))
        self.update()

    def update(self):
        self.o = self.o.apply(lambda s: s.map(
            lambda v: v.update() if isinstance(v, algebra.Vector) else v))


class FlatIndexField(Field):
    @abc.abstractmethod
    def replica_shifts(self, copies):
        """Return the sequence of scalars that perform the index translations"""

    @abc.abstractmethod
    def translate(self, series, z):
        """Return column `series` cyclically translated by scalar `z`"""


class R1Field(FlatIndexField):
    dtype = numpy.float64

    def __init__(self, start, end, cells, **kwargs):
        array, self.cell = numpy.linspace(start, end, cells, retstep=True)
        index = pandas.Index(numpy.around(array, INDEX_ROUND_TO)).astype(self.dtype)

        super().__init__(index=index, **kwargs)

    def __str__(self):
        return (f'ℝ-segment | cell {round(self.cell, 9)} with bounds {self.bounds}'
                f' | D = {self.dimension}')

    def scale_index(self, k):
        self.o.index *= k

    def translate(self, series, x):
        n = int(abs(x)/self.cell) % self.dimension

        if n == 0:
            return series

        if x < 0:
            n = self.dimension - n

        return pandas.Series(series.iloc[-n:].append(series.iloc[:-n]).to_numpy(),
                             index=self.o.index)

    def replicate_index(self, copies):
        return self.o.index.__class__(
            itertools.chain(*self.replicate(copies))).astype(self.dtype)

    def replica_shifts(self, copies):
        l, r = self.bounds

        for n in range(-copies, copies + 1):
            yield n*(r - l + self.cell)

    def replicate(self, copies):
        for shift in self.replica_shifts(copies):
            yield numpy.around(self.o.index + shift, INDEX_ROUND_TO)

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

    def scale_cell(self, k):
        self.cell *= k

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


class R2Field(FlatIndexField):
    dtype = numpy.complex128

    def __init__(self, sw, ne, re_cells, im_cells=None, **kwargs):
        im_cells = im_cells or re_cells
        x_array, self.x_cell = numpy.linspace(sw.real, ne.real, re_cells, retstep=True)
        y_array, self.y_cell = numpy.linspace(sw.imag, ne.imag, im_cells, retstep=True)
        self.x_array = numpy.around(x_array, INDEX_ROUND_TO)
        self.y_array = numpy.around(y_array, INDEX_ROUND_TO)

        super().__init__(index=self.index, **kwargs)

    def __str__(self):
        return (f'ℝ²-rectangle | cell {self.x_cell}x{self.y_cell} with '
                f'bounds {self.bounds} | D = {self.dimension}')

    @property
    def index(self):
        return pandas.Index([
            complex(i, j) for i, j in itertools.product(self.x_array, self.y_array)
        ]).astype(self.dtype)

    def as_frame(self, series):
        return pandas.Series(series.to_numpy(), index=pandas.MultiIndex.from_product(
            [self.x_array, self.y_array])).unstack(level=0)

    def as_series(self, frame):
        return pandas.Series(frame.transpose().stack().to_numpy(), index=self.o.index)

    @property
    def measure(self):
        return self.x_cell*self.y_cell

    @property
    def limits(self):
        (rmi, rma), (imi, ima) = self.bounds

        return ((rmi - self.x_cell/2, rma + self.x_cell/2),
                (imi - self.y_cell/2, ima + self.y_cell/2))

    @property
    def cell_limits(self):
        (rl, rr), (il, ir) = self.limits

        return (tuple(self.x_array - self.x_cell/2) + (rr,),
                tuple(self.y_array - self.y_cell/2) + (ir,))

    @property
    def bounds(self):
        return [[self.x_array.min(), self.x_array.max()],
                [self.y_array.min(), self.y_array.max()]]

    def scale_cell(self, k):
        self.x_cell *= k
        self.y_cell *= k

    def scale_index(self, k):
        self.o.index = pandas.Index([k*i for i in self.o.index])

    def replica_shifts(self, copies):
        X, Y = self.x_replica_shifts(copies), self.y_replica_shifts(copies)

        for x, y in itertools.product(X, Y):
            yield complex(x, y)

    def x_replica_shifts(self, copies):
        rmi, rma = self.bounds[0]

        for m in range(-copies, copies + 1):
            yield m*(rma - rmi + self.x_cell)

    def y_replica_shifts(self, copies):
        imi, ima = self.bounds[1]

        for n in range(-copies, copies + 1):
            yield n*(ima - imi + self.y_cell)

    def replicate_index(self, copies):
        X, Y = self.x_replica_shifts(copies), self.y_replica_shifts(copies)
        self.x_array, self.y_array = (
            numpy.around(numpy.concatenate(arrs), INDEX_ROUND_TO)
            for arrs in ([self.x_array + x for x in X], [self.y_array + y for y in Y]))

        return self.index

    def translate(self, series, z):
        frame = self.as_frame(series)
        m = int(abs(z.real)/self.x_cell) % len(self.x_array)
        n = int(abs(z.imag)/self.y_cell) % len(self.y_array)

        if m == n == 0:
            return series

        if z.real < 0:
            m = len(self.x_array) - m

        if z.imag < 0:
            n = len(self.y_array) - n

        frame = frame.iloc[-n:].append(frame.iloc[:-n]).transpose()
        frame = frame.iloc[-m:].append(frame.iloc[:-m]).transpose()

        return self.as_series(pandas.DataFrame(
            frame.to_numpy(), index=self.y_array, columns=self.x_array))

    def base_plot(self, axes, title=None):
        axes.set_title(title or str(self))
        xticks, yticks = self.cell_limits
        axes.xaxis.set_ticks(xticks, minor=True)
        axes.yaxis.set_ticks(yticks, minor=True)
        (rl, rr), (il, ir) = self.limits
        axes.set_xlim(rl, rr)
        axes.set_ylim(il, ir)
        axes.set_aspect('equal')
        axes.xaxis.set_label_text('X')
        axes.yaxis.set_label_text('Y')

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
        frame = self.as_frame(vector.image.density)
        xticks, yticks = self.cell_limits
        coll = axes.pcolormesh(xticks, yticks, frame.values, **kwargs)
        kwargs.pop('figure').colorbar(coll, ax=axes)
