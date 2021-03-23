import time
import unittest

from hilbert import stock


@stock.FrozenAttrs('foo', 'bar')
class MyType:
    def __init__(self, *args):
        self.foo, self.bar = args


class FrozenAttrsTests(unittest.TestCase):
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


class Klass(stock.Attr):
    def __init__(self, sleep_time, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.sleep_time = sleep_time
        self.lst = []

    @stock.Attr.getter
    def ab(self, a, b):
        time.sleep(self.sleep_time)
        return a + b

    @stock.Attr.getter
    def abc(self, ab, c):
        time.sleep(self.sleep_time)
        return ab * c

    @stock.Attr.getter
    def length(self, lst):
        time.sleep(self.sleep_time)
        return len(lst)

    @stock.Attr.mutates('lst')
    def append(self, x):
        self.lst.append(x)
        return self.lst


class AttrTests(unittest.TestCase):
    def setUp(self):
        self.obj = Klass(0.00001, a=2, b=-5, c=3)

    def test_set_and_del(self):
        self.assert_time_property('ab', -3, slow=True)
        self.assert_time_property('ab', -3, slow=False)
        self.assert_time_property('abc', -9, slow=True)
        self.assert_time_property('abc', -9, slow=False)
        self.obj.b = 0
        self.assert_time_property('abc', 6, slow=True)
        del self.obj.abc
        del self.obj.abc
        self.assert_time_property('abc', 6, slow=True)

    def test_set_defined_attr(self):
        self.assert_time_property('abc', -9, slow=True)
        self.obj.ab = 11
        assert not hasattr(self.obj, 'a')
        assert not hasattr(self.obj, 'b')
        self.assert_time_property('abc', 33, slow=True)
        self.assert_time_property('ab', 11, slow=False)
        self.obj.a = 7

        with self.assertRaises(AttributeError):
            self.obj.ab

        self.obj.b = 0
        self.assert_time_property('ab', 7, slow=True)
        self.assert_time_property('abc', 21, slow=True)

    def test_mutation_method(self):
        self.assert_time_property('length', 0, slow=True)
        self.assert_time_property('length', 0, slow=False)
        assert self.obj.append(22) == [22]
        self.assert_time_property('length', 1, slow=True)
        self.assert_time_property('length', 1, slow=False)

    def assert_time_property(self, name, expected_value, slow=False):
        t0 = time.time()
        value = getattr(self.obj, name)
        t = time.time() - t0

        assert value == expected_value

        if slow is True:
            assert t > self.obj.sleep_time
        else:
            assert t < self.obj.sleep_time


class ComplexIndexMixinTests(unittest.TestCase):
    def test_multiindex_at(self):
        index = pandas.MultiIndex.from_product([[1, 3, 5], [2, 4, 6]])

        assert all([stock.ComplexIndexMixin.multiindex_at(
            list(index.levels[0]).index(x), list(index.levels[1]).index(y), index
        ) == (x, y) for x, y in index])
