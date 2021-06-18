import jax
import jax.numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd


class Prior:
    def __init__(self, size: int, dist: tfd.Distribution = None) -> None:
        self._size = size
        self._dist = dist or tfd.Normal(loc=0.0, scale=1.0)
        self._seed = jax.random.PRNGKey(13)

    @property
    def dist(self) -> tfd.Distribution:
        return self._dist

    def sample(self) -> np.ndarray:
        return self._dist.sample(self._size, seed=self._seed)

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        return self._dist.log_prob(value)
