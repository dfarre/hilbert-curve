import unittest

import pandas

from hilbert import stock


@stock.FrozenLazyAttrs(('foo', 'bar'), ('prop', 'prap'))
class MyType:
    def __init__(self, *args):
        self.foo, self.bar = args

    def _make_prop(self):
        return 'prOp'

    def _make_prap(self):
        return 'prAp'


class FrozenLazyAttrsTests(unittest.TestCase):
    obj = MyType('Foo', 'Bar')

    def assert_reset_raises(self, key):
        self.assertRaisesRegex(
            stock.ImmutableReset,
            f"Immutable attribute '{key}' already set to {getattr(self.obj, key)}",
            setattr, self.obj, key, 'new')

    def test_frozen(self):
        assert all([hasattr(self.obj, f'_{key}') for key in ('foo', 'bar')])

        for key in ('foo', 'bar'):
            self.assert_reset_raises(key)

    def test_lazy_frozen(self):
        assert not any([hasattr(self.obj, f'_{key}') for key in ('prop', 'prap')])

        p, q = self.obj.prop, self.obj.prap

        assert (self.obj._prop, self.obj._prap) == (p, q)

        for key in ('prop', 'prap'):
            self.assert_reset_raises(key)


class ComplexIndexMixinTests(unittest.TestCase):
    def test_multiindex_at(self):
        index = pandas.MultiIndex.from_product([[1, 3, 5], [2, 4, 6]])

        assert all([stock.ComplexIndexMixin.multiindex_at(
            list(index.levels[0]).index(x), list(index.levels[1]).index(y), index
        ) == (x, y) for x, y in index])
