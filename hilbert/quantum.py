import abc
import math

import numpy
import pandas

from hilbert import EQ_ROUND_TO

from hilbert.algebra import PolarComplex as C
from hilbert import spaces
from hilbert import stock


class HistorySet(stock.WrappedDataFrame):
    def __init__(self, system, *args, description='History {label}', attr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system
        self.description = description
        self.attr = attr or {}

    def simulate(self, step=None, **kwargs):
        self.show(numpy.random.choice(self.o.columns, p=self.distribution),
                  step, **kwargs)

    def simulate_density(self, step=None, **kwargs):
        self.show_density(numpy.random.choice(self.o.columns, p=self.distribution),
                          step, **kwargs)

    def show(self, label, step=None, **kwargs):
        vectors = self.time_coarse(step)[label] if step else self.o[label]
        self.system.space.show_vectors(
            *vectors, title=self.description.format(label=label), **kwargs)

    def show_density(self, label, step=None, **kwargs):
        vectors = self.time_coarse(step)[label] if step else self.o[label]
        self.system.space.show_vectors_density(
            *vectors, title=self.description.format(label=label), **kwargs)

    def time_coarse(self, step: int):
        return self.o.reindex(index=[
            self.o.index[i] for i in numpy.arange(0, len(self.o), step)])

    @stock.PyplotShow()
    def show_distribution(self, **kwargs):
        axes = kwargs.pop('axes')
        self.distribution.plot(ax=axes, **kwargs)
        axes.yaxis.set_label_text('prob')
        axes.xaxis.set_label_text('history')
        axes.set_title(self.description.format(label='({0} ± {1})'.format(*self.mean_std)))
        axes.grid()

    @stock.Attr.getter
    def mean_std(self, distribution):
        p = distribution
        mean = round(p@p.index, EQ_ROUND_TO)

        return mean, round(numpy.sqrt(p@(p.index - mean)**2), EQ_ROUND_TO)

    @stock.Attr.getter
    def distribution(self, amplitudes, weight):
        return (amplitudes.abs()**2)/weight

    @stock.Attr.getter
    def weight(self, amplitudes):
        return (amplitudes.abs()**2).sum()

    @stock.Attr.getter
    def consistent(self, brakets, amplitudes):
        return not (brakets.to_numpy().real - numpy.diag(amplitudes.abs()**2)
                    ).round(EQ_ROUND_TO).any()

    @stock.Attr.getter
    def brakets(self, amplitudes):
        tf = self.o.index[-1]
        return pandas.DataFrame({b: pandas.Series([
            amplitudes[a].conjugate()*amplitudes[b]*(self[tf, a]@self[tf, b])
            for a in self.o.columns], index=self.o.columns) for b in self.o.columns})

    @stock.Attr.getter
    def amplitudes(self, chains):
        return chains.apply(lambda s: s.prod())

    @stock.Attr.getter
    def chains(self, o):
        return pandas.DataFrame({k: [o[k].iat[i]@(
            self.system(o.index[i], o.index[i-1])@(o[k].iat[i-1]))
            for i in range(1, len(o.index))] for k in o.columns
        }, index=o.index[1:])


class System(stock.Attr, metaclass=abc.ABCMeta):
    def __init__(self, space):
        super().__init__(space)
        self.space = space

    @abc.abstractmethod
    def __call__(self, tf, ti):
        """Return time development operator from `ti` to `tf`"""

    def __getitem__(self, ordinal):
        return self.space.spaces[ordinal]

    def unitary_history(self, vector, tf, ti=0, length=None, label=0):
        index = (numpy.arange(ti, tf, (tf - ti) // length if length else 1)
                 if isinstance(tf - ti, int) and (not length or not (tf - ti) % length)
                 else numpy.linspace(ti, tf, length))

        return HistorySet(self, {label: pandas.Series([
            self(t, ti)@vector for t in index], index=index)}, description='Unitary')

    def show_evolution(self, vector, tf, ti=0, ncurves=None, label=0, **kwargs):
        self.unitary_history(vector, tf, ti, ncurves, label).show(label, **kwargs)

    def show_density_evolution(self, vector, tf, ti=0, ncurves=None, label=0, **kwargs):
        self.unitary_history(vector, tf, ti, ncurves, label).show_density(label, **kwargs)

    def variance(self, operator, vector):
        return self.mean(operator@operator, vector) - self.mean(operator, vector)**2

    @staticmethod
    def mean(operator, vector):
        return (vector@(operator@vector))/(vector@vector)


class HamiltonianSystem(System):
    def __call__(self, ti, tf):
        return self.space.unitary_op(-self.hamiltonian*(tf - ti), validate=False)


class HoppingSystem(System):
    """Discrete-time system"""

    def __call__(self, tf, ti):
        return self.space.operator(
            numpy.linalg.matrix_power(self.hop_op.o.to_numpy(), tf - ti))

    def collapse(self, vector, *x):
        v = self.hop_op@(vector - sum(vector(ix)*self.space[ix, 'delta'] for ix in x))
        return v/v.norm

    def collapsing_histories(self, initial_vector, *x, go_on=lambda v: True):
        vectors = [initial_vector/initial_vector.norm]

        while go_on(vectors):
            vector = self.collapse(vectors[-1], *x)
            vectors.append(vector)
            weights = pandas.Series([vector(ix) for ix in x], index=x).abs()**2

            if round(weights.sum(), EQ_ROUND_TO) == 1:
                w = numpy.round(weights, EQ_ROUND_TO)
                points = list(w[w != 0].index)
                break
        else:
            return HistorySet(self, {len(vectors) - 1: pandas.Series(vectors)},
                              description='Escape at t = {label}')

        return HistorySet(self, {ix: pandas.Series(
            vectors + [self.hop_op@(self.space[ix, 'delta'])]) for ix in points
        }, description='Collapse at {label}')


class R1System(System):
    def __init__(self, lbound, rbound, dimension, **attr):
        super().__init__(spaces.R1LebesgueSpace(lbound, rbound, dimension), **attr)

    @stock.Attr.getter
    def position_op(self):
        return self.space.operator(numpy.diag(self.space.bases.domain()))

    @stock.Attr.getter
    def momentum_op(self):
        Pp = self.space.operator(numpy.diag(self.space.fourier_labels))
        F = self.space.fourier_op

        return F@Pp@F.dagger()


class R1C2System(System):
    def __init__(self, lbound, rbound, length, **attr):
        super().__init__(spaces.CurveSpaceProduct(
            spaces.CnSpace(2), spaces.R1LebesgueSpace(lbound, rbound, length)
        ), **attr)


class ToyTrain(R1System, HoppingSystem):
    @stock.Attr.getter
    def hop_op(self, space):
        return space.cycle_op

    def detection_histories(self, initial_vector, x):
        phi = self.collapsing_history(initial_vector, x).o[0]
        L = len(phi.index)

        return HistorySet(self, {tau: phi[:tau].append(pandas.Series(
            [self.space[x + 1 + i, 'delta'] for i in range(L - tau)], index=phi.index[tau:])
        ) for tau in range(1, L + 1)}, description='Detection at t = {label}')


class SplitToyTrain(R1C2System, HoppingSystem):
    @property
    def cell(self):
        return self[1].bases.cell

    def add_splitter(self, hop_op, x, theta=numpy.pi/4, phi=0):
        hop_op.put((1, x + self.cell), (1, x), math.cos(theta)*C(1, phi).number)
        hop_op.put((2, x + self.cell), (1, x), math.sin(theta))
        hop_op.put((1, x + self.cell), (2, x), -math.sin(theta)*C(1, phi).number)
        hop_op.put((2, x + self.cell), (2, x), math.cos(theta))

    def _make_hop_op(self):
        hop_op = self.space.cycle_op

        for kw in self.attr.get('splitters', ()):
            self.add_splitter(hop_op, **kw)

        return hop_op

    @classmethod
    def mach_zehnder(cls, left_arm, inner_arm, right_arm=None, split0kw=None, split1kw=None):
        right_arm = right_arm or left_arm
        length = left_arm + 1 + inner_arm + right_arm
        splitters = ({'x': 0, **(split0kw or {})}, {'x': inner_arm, **(split1kw or {})})

        return cls(-left_arm, inner_arm + right_arm, length, splitters=splitters)


class QuasiFreeParticleR1(R1System, HamiltonianSystem):
    @stock.Attr.getter
    def hamiltonian(self):
        Hp = self.space.operator(numpy.diag(self.space.fourier_labels**2))
        F = self.space.fourier_op

        return F@Hp@F.dagger()
