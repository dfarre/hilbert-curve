import abc
import functools

from matplotlib import pyplot


class Repr(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self):
        """Object's text content"""

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self}>'


class Eq(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eq(self, other):
        """Return self == other for same type"""

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.eq(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class Hashable(Eq, metaclass=abc.ABCMeta):
    def __hash__(self):
        return hash(self.eqkey())

    @abc.abstractmethod
    def eqkey(self):
        """Return hashable, frozen property to compare to others"""

    def eq(self, other):
        return self.eqkey() == other.eqkey()


class ImmutableReset(Exception):
    """Raised on reset try of immutable attributes"""


class FrozenLazyAttrs:
    def __init__(self, frozen_keys=(), lazy_keys=()):
        self.frozen_keys, self.lazy_keys = frozen_keys, lazy_keys

    def __call__(self, cls):
        self.init_keys(cls)
        cls.__init__ = self.decorate_init(cls.__init__)

        for key in self.frozen_keys:
            setattr(cls, key, property(
                functools.partial(self.get, key), functools.partial(self.set, key)))

        for key in self.lazy_keys:
            setattr(cls, key, property(
                functools.partial(self.setget, key), functools.partial(self.set, key)))

        return cls

    def init_keys(self, obj):
        self_frozen = dict.fromkeys(self.frozen_keys + self.lazy_keys, False)
        obj._frozen = {**getattr(obj, '_frozen', {}), **self_frozen}

    def decorate_init(self, function):
        def init(instance, *args, **kwargs):
            self.init_keys(instance)
            function(instance, *args, **kwargs)

        return init

    @staticmethod
    def set(key, instance, value):
        if instance._frozen[key] is True:
            raise ImmutableReset(
                f"Immutable attribute '{key}' already set to {getattr(instance, key)}")

        setattr(instance, f'_{key}', value)
        instance._frozen[key] = True

    @staticmethod
    def setget(key, instance):
        if instance._frozen[key] is False:
            value = getattr(instance, f'_make_{key}')()
            setattr(instance, f'_{key}', value)
            instance._frozen[key] = True

            return value

        return getattr(instance, f'_{key}')

    @staticmethod
    def get(key, instance):
        return getattr(instance, f'_{key}')


class IndexFrame(Repr, metaclass=abc.ABCMeta):
    """Pandas data frame wrapper for custom indexing with complex numbers"""
    def __init__(self, data_frame):
        self.o = data_frame

    def __str__(self):
        return f'\n{self.o}'

    @abc.abstractstaticmethod
    def to_index(x):
        """Map coordinate `x` to index label"""

    @abc.abstractstaticmethod
    def to_coordinate(label):
        """Map index label to coordinate"""

    def index_domain(self, index, copy=False):
        return index.map(self.to_coordinate).to_numpy(dtype=self.dtype, copy=copy)

    def domain(self, copy=False):
        return self.index_domain(self.o.index, copy)


class ComplexIndexMixin:
    @staticmethod
    def to_index(x):
        if x is not None:
            return x.real, x.imag

    @staticmethod
    def to_coordinate(label):
        return complex(*label)

    @staticmethod
    def multiindex_at(i, j, multiindex):
        return multiindex[len(multiindex.levels[1])*i + j]


class RealIndexMixin:
    @staticmethod
    def to_index(x):
        return x

    @staticmethod
    def to_coordinate(label):
        return label


class IndexXFrame(IndexFrame):
    def __getitem__(self, loc):
        return self.o.loc[self.location(*loc)]

    def __setitem__(self, loc, value):
        self.o.loc[self.location(*loc)] = value

    def location(self, xloc, column_loc):
        return ((slice(*map(self.to_index, (xloc.start, xloc.stop)))
                if isinstance(xloc, slice) else self.to_index(xloc)), column_loc)

    def at(self, x, column):
        return self.o.at[self.to_index(x), column]

    def setat(self, x, column, value):
        self.o.at[self.to_index(x), column] = value


class IndexXYFrame(IndexFrame):
    def __getitem__(self, loc):
        return self.o.loc[tuple(map(self.location, loc))]

    def __setitem__(self, loc, value):
        self.o.loc[tuple(map(self.location, loc))] = value

    def location(self, xloc):
        return (slice(*map(self.to_index, (xloc.start, xloc.stop)))
                if isinstance(xloc, slice) else self.to_index(xloc))

    def at(self, x, y):
        return self.o.at[self.to_index(x), self.to_index(y)]

    def setat(self, x, y, value):
        self.o.at[self.to_index(x), self.to_index(y)] = value


class PyplotShow:
    golden = 1.61803398875

    def __init__(self, **fig_kwds):
        self.fig_kwds = fig_kwds
        self.fig_kwds.setdefault('figsize', (2*self.golden*6, 6))
        self.fig_kwds.setdefault('facecolor', 'w')

    def __call__(self, plot_method):
        @functools.wraps(plot_method)
        def wrapper(obj, *args, fig_kwds=None, **kwargs):
            figure, axes = pyplot.subplots(**{**self.fig_kwds, **(fig_kwds or {})})
            plot_method(obj, *args, figure=figure, axes=axes, **kwargs)
            pyplot.show()

            return figure, axes

        return wrapper
