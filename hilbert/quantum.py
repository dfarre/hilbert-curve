import abc

import numpy

from hilbert import spaces
from hilbert import stock


@stock.FrozenLazyAttrs(('space',))
class System(metaclass=abc.ABCMeta):
    def __init__(self, space):
        self.space = space

    @abc.abstractmethod
    def __call__(self, tf, ti):
        """Return time development operator from `ti` to `tf`"""

    def evolve(self, tf, ti, vector):
        return self(tf, ti)@vector

    def show_evolution(self, tf, ti, vector, ncurves, **kwargs):
        self.space.show_vectors(
            *(self.evolve(t, ti, vector) for t in numpy.linspace(ti, tf, ncurves)),
            **kwargs)

    def show_density_evolution(self, tf, ti, vector, ncurves, **kwargs):
        self.space.show_vectors_density(
            *(self.evolve(t, ti, vector) for t in numpy.linspace(ti, tf, ncurves)),
            **kwargs)

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
        super().__init__(spaces.R1Field.range(
            spaces.LebesgueCurveSpace, lbound, rbound, dimension))

    @property
    def position_op(self):
        return self.space.operator(numpy.diag(self.space.bases.domain()))

    @property
    def momentum_op(self):
        Pp = self.space.operator(numpy.diag(self.space.fourier_labels))
        F = self.space.fourier_op

        return F@Pp@F.dagger()


class QuasiFreeParticleR1(R1System, HamiltonianSystem):
    def _make_hamiltonian(self):
        Hp = self.space.operator(numpy.diag(self.space.fourier_labels**2))
        F = self.space.fourier_op

        return F@Hp@F.dagger()
