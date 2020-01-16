import abc

import numpy
import pandas

from hilbert import EQ_ROUND_TO

from hilbert.algebra import PolarComplex as C
from hilbert import spaces
from hilbert import stock


@stock.FrozenLazyAttrs(lazy_keys=('amplitudes', 'brakets', 'consistent'))
class HistorySet(stock.WrappedDataFrame):
    def __init__(self, system, *args, description='History {label}', **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system
        self.description = description

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

    @property
    def mean_std(self):
        p = self.distribution
        mean = round(p@p.index, EQ_ROUND_TO)

        return mean, round(numpy.sqrt(p@(p.index - mean)**2), EQ_ROUND_TO)

    @property
    def distribution(self):
        return (self.amplitudes.abs()**2)/self.weight

    @property
    def weight(self):
        return (self.amplitudes.abs()**2).sum()

    def _make_consistent(self):
        return not (self.brakets.to_numpy().real - numpy.diag(self.amplitudes.abs()**2)
                    ).round(EQ_ROUND_TO).any()

    def _make_brakets(self):
        tf = self.o.index[-1]
        return pandas.DataFrame({b: pandas.Series([
            self.amplitudes[a].conjugate()*self.amplitudes[b]*(self[tf, a]@self[tf, b])
            for a in self.o.columns], index=self.o.columns) for b in self.o.columns})

    def _make_amplitudes(self):
        return pandas.Series([numpy.product([self.o[k].iat[i]@(
            self.system(self.o.index[i], self.o.index[i-1])@(self.o[k].iat[i-1]))
            for i in range(1, len(self.o.index))]) for k in self.o.columns
        ], index=self.o.columns)


@stock.FrozenLazyAttrs(('space',))
class System(metaclass=abc.ABCMeta):
    def __init__(self, space, **attr):
        self.space, self.attr = space, attr

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


@stock.FrozenLazyAttrs(lazy_keys=('hop_op',))
class HoppingSystem(System):
    """Discrete-time system"""

    def __call__(self, tf, ti):
        return self.space.operator(
            numpy.linalg.matrix_power(self.hop_op.o.to_numpy(), tf - ti))

    def collapse(self, vector, *x):
        v = self.hop_op@(vector - sum(vector(ix)*self.space[ix, 'delta'] for ix in x))

        if round(v.norm, EQ_ROUND_TO):
            return v/v.norm

    def collapsing_history(self, initial_vector, *x, go_on=lambda v: True, label=0):
        vectors = [initial_vector/initial_vector.norm]

        while go_on(vectors):
            vector = self.collapse(vectors[-1], *x)

            if vector is not None:
                vectors.append(vector)

                if any(round(abs(vector(ix)), EQ_ROUND_TO) == 1 for ix in x):
                    description = 'Collapse'
                    break
            else:
                description = 'Collapse'
                break
        else:
            description = 'Escape'

        return HistorySet(self, {label: pandas.Series(vectors)}, description=description)


@stock.FrozenLazyAttrs(lazy_keys=('hamiltonian',))
class HamiltonianSystem(System):
    def __call__(self, ti, tf):
        return self.space.unitary_op(-self.hamiltonian*(tf - ti), validate=False)


class R1System(System):
    def __init__(self, lbound, rbound, dimension, **attr):
        super().__init__(spaces.R1LebesgueSpace(lbound, rbound, dimension), **attr)

    @property
    def position_op(self):
        return self.space.operator(numpy.diag(self.space.bases.domain()))

    @property
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
    def _make_hop_op(self):
        return self.space.cycle_op

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

    def add_splitter(self, hop_op, x):
        coef = 1/numpy.sqrt(2)
        hop_op.put((1, x + self.cell), (1, x), coef)
        hop_op.put((2, x + self.cell), (1, x), coef)
        hop_op.put((1, x + self.cell), (2, x), -coef)
        hop_op.put((2, x + self.cell), (2, x), coef)

    def add_phase_shift(self, hop_op, ix, phase):
        hop_op.put((ix[0], ix[1] + self.cell), ix, C(1, phase).number)

    def _make_hop_op(self):
        hop_op = self.space.cycle_op

        for x in self.attr.get('splitters', ()):
            self.add_splitter(hop_op, x)

        for ix, phase in self.attr.get('phases', {}).items():
            self.add_phase_shift(hop_op, ix, phase)

        return hop_op

    @classmethod
    def mach_zehnder(cls, left_arm, inner_arm, right_arm=None, phases=(0, 0)):
        right_arm = right_arm or left_arm
        center = int(inner_arm/2)
        length = left_arm + 1 + inner_arm + right_arm

        return cls(-left_arm, inner_arm + right_arm, length,
                   splitters=(0, inner_arm), phases={(1, center): phases[0],
                                                     (2, center): phases[1]})


class QuasiFreeParticleR1(R1System, HamiltonianSystem):
    def _make_hamiltonian(self):
        Hp = self.space.operator(numpy.diag(self.space.fourier_labels**2))
        F = self.space.fourier_op

        return F@Hp@F.dagger()
