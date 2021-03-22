import abc
import collections
import functools
import inspect
<<<<<<< HEAD

import pandas
=======
>>>>>>> master

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


class FrozenAttrs:
    def __init__(self, *frozen_keys):
        self.frozen_keys = frozen_keys

    def __call__(self, cls):
        self.init_keys(cls)
        cls.__init__ = self.decorate_init(cls.__init__)

        for key in self.frozen_keys:
            setattr(cls, key, property(
                functools.partial(self.get, key), functools.partial(self.set, key)))

        return cls

    def init_keys(self, obj):
        self_frozen = dict.fromkeys(self.frozen_keys, False)
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
    def get(key, instance):
        return getattr(instance, f'_{key}')


class Attr:
    _marked_getters = collections.defaultdict(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attr_cache = {}

    @classmethod
    def __init_subclass__(cls):
        cls._attr_tree = collections.defaultdict(set)
        cls._attr_targets = collections.defaultdict(set)
        cls_path = f'{cls.__module__}.{cls.__name__}'

        for method in Attr._marked_getters[cls_path]:
            setattr(cls, method.__name__, property(cls._make_getter(method)))

        del Attr._marked_getters[cls_path]

    @classmethod
    def _make_getter(cls, method):
        _, *arg_names = inspect.getargspec(method)[0]

        if arg_names:
            cls._attr_tree[method.__name__].update(set(arg_names) - set(cls._attr_tree))

            for key in set(arg_names) & set(cls._attr_tree):
                cls._attr_tree[method.__name__].update(cls._attr_tree[key])

            for key in cls._attr_tree[method.__name__]:
                cls._attr_targets[key].add(method.__name__)

        @functools.wraps(method)
        def getter_method(self):
            if method.__name__ in self._attr_cache:
                return self._attr_cache[method.__name__]

            value = method(self, *(getattr(self, name) for name in arg_names))
            self._attr_cache[method.__name__] = value

            return value

        return getter_method

    @classmethod
    def getter(cls, method):
        caller_stack = inspect.stack()[1]
        module_name = inspect.getmodule(caller_stack.frame).__name__
        cls._marked_getters[f'{module_name}.{caller_stack.function}'].append(method)
        return method

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self.reset(key)

    def __delattr__(self, key):
        if key not in self._attr_tree:
            super().__delattr__(key)
        elif key in self._attr_cache:
            del self._attr_cache[key]

    def reset(self, key):
        if key in self._attr_targets:
            for target_key in self._attr_targets[key] & set(self._attr_cache):
                del self._attr_cache[target_key]


class WrappedDataFrame(Repr):
    """Pandas data frame wrapper - an alternative to subclassing"""
    dtype = None

    def __init__(self, *args, **kwargs):
        self.o = pandas.DataFrame(*args, **kwargs)

    def __str__(self):
        return f'\n{self.o}'

    def __getitem__(self, location):
        return self.o.loc[location]

    def __setitem__(self, location, value):
        self.o.loc[location] = value

    def at(self, x, column):
        return self.o.at[x, column]

    def put(self, x, column, value):
        if x not in self.o.index:
            raise KeyError(f'{x} not in index')

        self.o.at[x, column] = value

    def index_domain(self, index, copy=False):
        return index.to_numpy(dtype=self.dtype, copy=copy)

    def domain(self, copy=False):
        return self.index_domain(self.o.index, copy=copy)


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


def index_from_product(*indices):
    levels = []

    for index in indices:
        if isinstance(index, pandas.MultiIndex):
            levels.extend(index.levels)
        else:
            levels.append(index)

    return pandas.MultiIndex.from_product(levels)
