import abc

import pandas
import numpy

from hilbert import EQ_ROUND_TO, INDEX_ROUND_TO

from hilbert import algebra
from hilbert import fields
from hilbert import operators
from hilbert import stock

from scipy import linalg

from hilbert.curves import lib
from hilbert.curves import vectors


class InvalidVectorMade(Exception):
    """Raised for wrong vector input"""


class Space(stock.Repr, stock.Attr, metaclass=abc.ABCMeta):
    def __init__(self, bases: fields.FlatIndexField, validate_basis=False):
        super().__init__()

        self.bases, self.validate_basis = bases, validate_basis

    def __str__(self):
        return repr(self.bases)

    @property
    def dimension(self):
        return self.bases.dimension

    @property
    def index(self):
        return self.bases.o.index

    @property
    def domain(self):
        return self.bases.domain()

    @property
    def measure(self):
        return self.bases.measure

    @property
    def spaces(self):
        return [self]

    @property
    def Id(self):
        return self.operator(numpy.identity(self.dimension))

    @property
    def cycle_op(self):
        cols = list(self.Id.o)
        return self.operator(self.Id.o[cols[1:]+cols[0:1]].to_numpy())

    def operator(self, *args, **kwargs):
        return operators.Operator(
            self, *args, columns=self.index, index=self.index, **kwargs)

    def __call__(self, *args, validate=True, **kwargs):
        vector = self.make_vector(*args, **kwargs)

        if validate:
            self.validate(vector)

        return vector

    def __setitem__(self, key, basis):
        """Set basis"""
        self.bases.o[key] = basis

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
                    self.bases.o.at[ix, key] = def_method(ix)

                return self.bases[xloc, key]
        elif not isinstance(vec, algebra.Vector):
            def_method = self.try_def_method(key)
            vector = def_method(xloc)
            self.bases.put(xloc, key, vector)

            return vector

        return vec

    def try_def_method(self, key):
        def_method = getattr(self, f'def_{key}_vector', None)

        if def_method is None:
            raise KeyError(f"Requested new vector(s) for non-analytic basis '{key}'")

        return def_method

    @abc.abstractmethod
    def make_vector(self, *args, series=None, **kwargs):
        """Vector instance constructor returning new vector"""

    @abc.abstractmethod
    def scale_basis_vector(self, k, vector):
        """Keep norm invariant"""

    @staticmethod
    def validate(vector):
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
        analytic = list(filter(lambda key: self.is_analytic(key), self.bases.o))
        non_analytic = set(self.bases.o) - set(analytic)

        if non_analytic:
            shifts = list(self.bases.replica_shifts(copies))

        self.bases.extend_index(copies)

        for key in non_analytic:
            for z, v in self.bases.o[key].dropna().items():
                for shift in shifts:
                    c = z + shift
                    ix = (complex(round(c.real, INDEX_ROUND_TO), round(c.imag, INDEX_ROUND_TO))
                          if isinstance(c, complex) else round(c, INDEX_ROUND_TO))
                    self.bases.put(ix, key, self.translate(v, shift))

        for key in analytic:
            getattr(self, f'extend_{key}_basis')(copies)

    def translate(self, vector, shift):
        return self(series=self.bases.translate(vector.image.i, shift))

    def op_from_callable(self, function, *apply_args, **apply_kwds):
        return self.operator().apply(function, *apply_args, **apply_kwds)

    def unitary_op(self, hermitian, validate=True):
        if validate and not hermitian.is_hermitian():
            raise NotImplementedError('An Hermitian generator is required')

        H = (hermitian if not hermitian.is_polar() else hermitian.toggle_polar())

        return self.operator(linalg.expm(1j*H.o.to_numpy()))

    @stock.Attr.getter
    def fourier_op(self, bases):
        if not (self.operator_type == operators.ROperator and bases.dimension % 2 == 0):
            raise NotImplementedError('Fourier basis is defined over ℝ with even dimension')

        return self.op_from_callable(self.fourier_basis_coords, axis=1)

    @stock.Attr.getter
    def fourier_labels(self, bases):
        return pandas.Series(self.fourier_label(bases.domain()),
                             index=bases.o.index, name='p')

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

    @stock.PyplotShow()
    def show_vectors_density(self, *vectors, **kwargs):
        self.plot_vectors_density(vectors, kwargs.pop('axes'), **kwargs)

    @staticmethod
    def plot_vectors(vectors, top_ax, bottom_ax, title=None, **kwargs):
        for vector in vectors:
            vector.image.full_plot(top_ax, bottom_ax, **kwargs)

        top_ax.yaxis.set_label_text('Re')
        bottom_ax.yaxis.set_label_text('Im')
        top_ax.grid(), bottom_ax.grid()

        if title is not None:
            top_ax.set_title(title)

        return top_ax, bottom_ax

    @staticmethod
    def plot_vectors_density(vectors, axes, **kwargs):
        for vector in vectors:
            vector.image.density_plot(axes, **kwargs)

        axes.yaxis.set_label_text('abs²')
        axes.grid()

        return axes


class LebesgueCurveSpace(Space):
    def def_delta_vector(self, x):
        return self(lib.Delta(1/numpy.sqrt(self.bases.measure), pole=x),
                    validate=self.validate_basis)

    def def_fourier_vector(self, x):
        return self(lib.Exp(1/numpy.sqrt(self.bases.dimension*self.bases.measure),
                            -1j*self.fourier_label(x)), validate=self.validate_basis)

    def make_vector(self, *args, series=None, **kwargs):
        if series is None:
            return vectors.CurveVector(self, *args, **kwargs)

        return vectors.CurveVector(self, image=algebra.Image(
            series=series.reindex(index=self.index, fill_value=0)))

    def scale_basis_vector(self, k, vector):
        return vector.__class__(self, *(
            c.scale(self.norm_scale_factor(k), k) for c in vector.curves))

    def norm_scale_factor(self, k):
        return {numpy.float64: 1/numpy.sqrt(k), numpy.complex128: 1/k}[self.bases.dtype]

    def extend_delta_basis(self, copies):
        """Nothing to do - ready to be extended on demand"""

    def extend_fourier_basis(self, copies):
        p_series = self.fourier_labels
        vectors = self.bases.o['fourier'].dropna().to_numpy()
        self['fourier'] = numpy.nan

        for vector in vectors:
            p = -vector.curves[0].parameters[1].imag
            new_x = p_series[(p_series - p).round(EQ_ROUND_TO) == 0].index[0]
            self.bases.put(new_x, 'fourier', vector/numpy.sqrt(2*copies + 1))


class CurveSpaceProduct(LebesgueCurveSpace):
    def __init__(self, *spaces):
        self._spaces = spaces

    @property
    def spaces(self):
        return self._spaces


class Field(stock.Attr, stock.IndexXFrame, metaclass=abc.ABCMeta):
    dtype = None

    @stock.Attr.getter
    def dimension(self, o):
        return len(o.index)

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
        super().__init__(pandas.DataFrame(*args, **kwargs))
        self.cell = cell

    def __str__(self):
        return repr(self.spaces)

    def __getitem__(self, loc):
        """Get a product of basis vectors"""
        ix, keys = loc

        if isinstance(keys, str):
            keys = (keys,)*len(self.spaces)

        return algebra.TensorProduct(*(s[x, k] for s, x, k in zip(self.spaces, ix, keys)))

    @property
    def measure(self):
        return numpy.prod([space.measure for space in self.spaces])

    @property
    def dimension(self):
        return numpy.prod([space.dimension for space in self.spaces])

    @property
    def index(self):
        return stock.index_from_product(*(s.bases.o.index for s in self.spaces))


class R2LebesgueSpace(LebesgueCurveSpace):
    def __init__(self, sw, ne, re_cells, im_cells=None, validate_basis=False, **kwargs):
        super().__init__(fields.R2Field(sw, ne, re_cells, im_cells=im_cells, **kwargs),
                         validate_basis)


class R1LebesgueSpace(LebesgueCurveSpace):
    def __init__(self, start, end, cells, validate_basis=False, **kwargs):
        super().__init__(fields.R1Field(start, end, cells, **kwargs), validate_basis)


class CnSpace(R1LebesgueSpace):
    def __init__(self, dimension, validate_basis=False, **kwargs):
        super().__init__(1, dimension, dimension,  validate_basis, **kwargs)
