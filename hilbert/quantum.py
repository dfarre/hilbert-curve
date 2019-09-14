import abc

import numpy
import pandas

from hilbert import EQ_ROUND_TO

from hilbert import fields
from hilbert import spaces
from hilbert import stock


@stock.FrozenLazyAttrs(lazy_keys=('amplitudes', 'brakets', 'consistent'))
class HistorySet(stock.RealIndexMixin, stock.IndexXYFrame):
    def __init__(self, data_frame, system, description='History {label}'):
        super().__init__(data_frame)
        self.system = system
        self.description = description

    def simulate(self, **kwargs):
        self.show(numpy.random.choice(self.o.index, p=self.distribution), **kwargs)

    def simulate_density(self, **kwargs):
        self.show_density(numpy.random.choice(self.o.index, p=self.distribution), **kwargs)

    def show(self, label, **kwargs):
        self.system.space.show_vectors(
            *self[:, label], title=self.description.format(label=label), **kwargs)

    def show_density(self, label, **kwargs):
        self.system.space.show_vectors_density(
            *self[:, label], title=self.description.format(label=label), **kwargs)

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
    def __init__(self, space):
        self.space = space

    @abc.abstractmethod
    def __call__(self, tf, ti):
        """Return time development operator from `ti` to `tf`"""

    def unitary_history(self, tf, ti, vector, length, label=0):
        index = numpy.linspace(ti, tf, length)
        return HistorySet(pandas.DataFrame({label: pandas.Series([
            self(t, ti)@vector for t in index], index=index)}), self, 'Unitary {label}')

    def show_evolution(self, tf, ti, vector, ncurves, label=0, **kwargs):
        self.unitary_history(tf, ti, vector, ncurves, label).show(label, **kwargs)

    def show_density_evolution(self, tf, ti, vector, ncurves, label=0, **kwargs):
        self.unitary_history(
            tf, ti, vector, ncurves, label).show_density(label, **kwargs)

    def variance(self, operator, vector):
        return self.mean(operator@operator, vector) - self.mean(operator, vector)**2

    @staticmethod
    def mean(operator, vector):
        return (vector@(operator@vector))/(vector@vector)


@stock.FrozenLazyAttrs(lazy_keys=('hamiltonian',))
class HamiltonianSystem(System):
    def __call__(self, ti, tf):
        return self.space.unitary_op(-self.hamiltonian*(tf - ti), validate=False)


class R1System(System):
    def __init__(self, lbound, rbound, dimension):
        super().__init__(fields.R1Field.range(
            spaces.LebesgueCurveSpace, lbound, rbound, dimension))

    @property
    def position_op(self):
        return self.space.operator(numpy.diag(self.space.bases.domain()))

    @property
    def momentum_op(self):
        Pp = self.space.operator(numpy.diag(self.space.fourier_labels))
        F = self.space.fourier_op

        return F@Pp@F.dagger()


@stock.FrozenLazyAttrs(lazy_keys=('hop_op',))
class ToyTrain(R1System):
    def __call__(self, tf, ti):
        return self.space.operator(
            numpy.linalg.matrix_power(self.hop_op.o.to_numpy(), tf - ti))

    def _make_hop_op(self):
        cols = list(self.space.Id.o)
        return self.space.operator(self.space.Id.o[cols[1:]+cols[0:1]].to_numpy())

    def collapse(self, vector, x):
        v = self.hop_op@(vector - vector(x)*self.space[x, 'delta'])

        if round(v.norm, EQ_ROUND_TO):
            return v/v.norm

    def collapsing_history(self, initial_vector, x):
        vectors = [initial_vector/initial_vector.norm]

        while True:
            vector = self.collapse(vectors[-1], x)

            if vector is not None:
                vectors.append(vector)
            else:
                break

        return pandas.Series(vectors)

    def detection_histories(self, initial_vector, x):
        phi = self.collapsing_history(initial_vector, x)
        L = len(phi.index)

        return HistorySet(pandas.DataFrame({tau: phi[:tau].append(pandas.Series(
            [self.space[x + 1 + i, 'delta'] for i in range(L - tau)], index=phi.index[tau:])
        ) for tau in range(1, L + 1)}), self, 'Detection at t = {label}')


class QuasiFreeParticleR1(R1System, HamiltonianSystem):
    def _make_hamiltonian(self):
        Hp = self.space.operator(numpy.diag(self.space.fourier_labels**2))
        F = self.space.fourier_op

        return F@Hp@F.dagger()
