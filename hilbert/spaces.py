import abc
import itertools

import pandas
import numpy

from hilbert import EQ_ROUND_TO, INDEX_ROUND_TO

from hilbert import algebra
from hilbert import operators
from hilbert import stock

from scipy import linalg

from hilbert.curves import lib
from hilbert.curves import vectors


class InvalidVectorMade(Exception):
    """Raised for wrong vector input"""


@stock.FrozenLazyAttrs(('bases',))
class Space(stock.Repr, metaclass=abc.ABCMeta):
    def __init__(self, bases, validate_basis=False):
        self.operator_type, self.image_type = (
            (operators.COperator, vectors.CImage) if isinstance(bases, C1Field) else
            (operators.ROperator, vectors.RImage) if isinstance(bases, R1Field) else
            (None, None))

        if self.operator_type is None:
            raise NotImplementedError('`bases` argument should be a `Field`')

        self.bases, self.validate_basis = bases, validate_basis

    def __str__(self):
        return repr(self.bases)

    @property
    def Id(self):
        return self.operator_type(pandas.DataFrame(
            numpy.identity(self.bases.dimension),
            columns=self.bases.o.index, index=self.bases.o.index), self)

    def __call__(self, *args, validate=True, **kwargs):
        vector = self.make_vector(*args, **kwargs)

        if validate:
            self.validate(vector)

        return vector

    def __getitem__(self, loc):
        """Lazy basis vector definition using `loc` indexing"""
        xloc, key = loc

        if not isinstance(key, str):
            raise NotImplementedError('A single basis should be specified')

        if key not in self.bases.o:
            self.bases.o[key] = numpy.nan

        vec = self.bases[xloc, key]

        if isinstance(vec, pandas.Series):
            nans_at = vec[vec.map(lambda v: not isinstance(v, algebra.Vector))].index

            if len(nans_at):
                def_method = self.try_def_method(key)

                for ix in nans_at:
                    self.bases.o.at[ix, key] = def_method(self.bases.to_coordinate(ix))

                return self.bases[xloc, key]
        elif not isinstance(vec, algebra.Vector):
            def_method = self.try_def_method(key)
            vector = def_method(xloc)
            self.bases.setat(xloc, key, vector)

            return vector

        return vec

    def try_def_method(self, key):
        def_method = getattr(self, f'def_{key}_vector', None)

        if def_method is None:
            raise KeyError(f"Requested new vector(s) for non-analytic basis '{key}'")

        return def_method

    def __setitem__(self, key, basis):
        """Set basis"""
        self.bases.o[key] = basis

    @abc.abstractmethod
    def make_vector(self, *args, **kwargs):
        """Vector instance constructor returning new vector"""

    @abc.abstractmethod
    def scale_basis_vector(self, k, vector):
        """Keep norm invariant"""

    def validate(self, vector):
        braket = vector @ vector

        if not (braket > 0 and numpy.isfinite(braket)):
            raise InvalidVectorMade(
                f'{vector} does not belong to the space - no finite norm!')

    def scale(self, k):
        self.bases.scale_cell(k)
        self.bases.scale_index(k)

        for key in self.bases.o:
            self[key] = self.bases.o[key].map(
                lambda vector: self.scale_basis_vector(k, vector)
                if isinstance(vector, algebra.Vector) else vector)

    def extend(self, copies=1):
        replicas = self.bases.replicas(copies)
        vector_replicas = {key: list(
            self.bases.yield_vector_replicas(key, copies, replicas))
            for key in filter(lambda key: not self.is_analytic(key), self.bases.o)}
        self.bases.extend_index(copies, replicas)
        self.bases.update()

        for key in set(self.bases.o) - set(vector_replicas):
            getattr(self, f'extend_{key}_basis')(copies)

        for key, repls in vector_replicas.items():
            for z, series in repls:
                self.bases.setat(z, key, self(series=series, validate=False))

    def operator(self, *args, **kwargs):
        return self.operator_type(pandas.DataFrame(
            *args, columns=self.bases.o.index, index=self.bases.o.index, **kwargs
        ), self)

    def op_from_callable(self, function, *apply_args, **apply_kwds):
        return self.operator().apply(function, *apply_args, **apply_kwds)

    def unitary_op(self, hermitian):
        if not hermitian.is_hermitian():
            raise NotImplementedError('Operator.unitary requires an hermitian generator')

        H = (hermitian if not hermitian.is_polar() else hermitian.toggle_polar())

        return self.operator(linalg.expm(1j*H.o.to_numpy()))

    @property
    def fourier_op(self):
        if not (self.operator_type == operators.ROperator and self.bases.dimension % 2 == 0):
            raise NotImplementedError('Fourier basis is defined over ℝ with even dimension')

        return self.op_from_callable(self.fourier_basis_coords, axis=1)

    @property
    def fourier_labels(self):
        return pandas.Series(self.fourier_label(self.bases.domain()),
                             index=self.bases.o.index, name='p')

    def fourier_label(self, x):
        return 2*numpy.pi*(
            (x - self.bases.bounds[0])/(self.bases.measure*self.bases.dimension) - 0.5
        )/self.bases.measure

    def fourier_basis_coords(self, row):
        return self.fourier_labels.map(
            lambda p: numpy.exp(-1j*p*row.name)/numpy.sqrt(self.bases.dimension))

    def map_basis(self, operator, new_key='new', key='delta'):
        if not operator.space == self:
            raise NotImplementedError('Operator should belong to the space')

        self[new_key] = self[:, key].map(lambda v: operator@v)

    def is_orthonormal(self, key):
        basis = self[:, key]
        brakets = self.op_from_callable(lambda c: basis.map(lambda v: basis[c.name]@v))

        return brakets == self.Id

    def is_basis(self, key):
        return all([u == self.vector(key, self.coords(key, u)) for u in self[:, 'delta']])

    def is_analytic(self, key):
        return hasattr(self, f'def_{key}_vector') and hasattr(self, f'extend_{key}_basis')

    def coords(self, key, vector):
        return self[:, key].map(lambda v: v@vector)

    def vector(self, key, coords):
        return sum(coord*vector for coord, vector in zip(coords, self[:, key]))

    @stock.PyplotShow(nrows=2, ncols=1)
    def show_basis_slice(self, key, from_x=None, to_x=None, **kwargs):
        top_ax, _ = self.plot_vectors(
            self[from_x:to_x, key], *kwargs.pop('axes'), **kwargs)

        if hasattr(self, f'{key}_labels'):
            labels = getattr(self, f'{key}_labels')
            fr = '' if from_x is None else labels[from_x]
            to = '' if to_x is None else labels[to_x]
            name = labels.name
        else:
            fr, to = '' if from_x is None else from_x, '' if to_x is None else to_x
            name = 'x'

        top_ax.set_title(f'{key} basis ∀{name} ∈ ({fr}...{to})')

    @stock.PyplotShow(nrows=2, ncols=1)
    def show_vectors(self, *vectors, **kwargs):
        self.plot_vectors(vectors, *kwargs.pop('axes'), **kwargs)

    @staticmethod
    def plot_vectors(vectors, top_ax, bottom_ax, **kwargs):
        for v in vectors:
            v.full_plot(top_ax, bottom_ax, **kwargs)

        top_ax.yaxis.set_label_text('Re')
        bottom_ax.yaxis.set_label_text('Im')
        top_ax.grid(), bottom_ax.grid()

        return top_ax, bottom_ax


class LebesgueCurveSpace(Space):
    def def_delta_vector(self, x):
        return self(lib.Delta(1/numpy.sqrt(self.bases.measure), pole=x),
                    validate=self.validate_basis)

    def def_fourier_vector(self, x):
        return self(lib.Exp(1/numpy.sqrt(self.bases.dimension*self.bases.measure),
                            -1j*self.fourier_label(x)), validate=self.validate_basis)

    def make_vector(self, *args, series=None, **kwargs):
        if series is None:
            return vectors.Vector(self, *args, **kwargs)

        return vectors.Vector(self, lib.ImageCurve(self.image_type(
            series=series.reindex(index=self.bases.o.index, fill_value=0))))

    def scale_basis_vector(self, k, vector):
        return vector.__class__(self, *(
            c.scale(self.norm_scale_factor(k), k) for c in vector.curves))

    def norm_scale_factor(self, k):
        return {operators.ROperator: 1/numpy.sqrt(k),
                operators.COperator: 1/k}[self.operator_type]

    def extend_delta_basis(self, copies):
        """Nothing to do - ready to be extended on demand"""

    def extend_fourier_basis(self, copies):
        p_series = self.fourier_labels
        vectors = self.bases.o['fourier'].dropna().to_numpy()
        self['fourier'] = numpy.nan

        for vector in vectors:
            p = -vector.curves[0].parameters[1].imag
            new_x = p_series[(p_series - p).round(EQ_ROUND_TO) == 0].index[0]
            self.bases.setat(new_x, 'fourier', vector/numpy.sqrt(2*copies + 1))


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
        """Show domain"""

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
        values = vector.image.i.abs().unstack(level=0)
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
