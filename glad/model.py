from typing import List, Callable, Dict
import jax
import math
import numpy as np
import jax.numpy as jnp
from .prior import Prior
from loguru import logger
from functools import partial
from dataclasses import dataclass
from einops import repeat, reduce
from alive_progress import alive_bar
from jax.experimental import optimizers
import tensorflow_probability.substrates.jax.distributions as tfd


@dataclass
class Optim:
    init: Callable
    update: Callable
    get_params: Callable
    state: optimizers.OptimizerState = None
    _step: int = 0

    def params(self) -> Dict[str, jnp.ndarray]:
        return self.get_params(self.state) if self.state is not None else None

    def init_state_with(self, params: Dict[str, jnp.ndarray]) -> None:
        self.state = self.init(params)

    def step(self, grad: Dict[str, jnp.ndarray]) -> None:
        self.state = self.update(self._step, grad, self.state)
        self._step += 1


# TODO: Добавить partial fit, чтобы было обучение по батчам.
# Главная сложность - нужно будет переписать m_step_loss_fn,
# чтобы inner_data был одним из аргументов на входе,
# а то иначе для него не будет jit, если state меняться будет.
# Альтернатива - сделать из inner_data pytree как во flax.
# Ещё нужно подумать над кейсом, когда можно детектить случай,
# в котором модель просто инвертирует предсказания, такое достаточно часто случается.

# GLAD requires answers to start from 0.
class GLAD:
    def __init__(
        self,
        num_tasks: int,
        num_labelers: int,
        num_classes: int = 2,
        learning_rate: float = 0.001,
        alpha_prior: Prior = None,
        log_beta_prior: Prior = None,
        tol: float = 1e-4,
        max_steps: int = 1000,
        grad_steps: int = 1,
        seed: int = 13,
    ) -> None:
        self._learning_rate = learning_rate
        self._num_classes = num_classes
        self._tol = tol
        self._max_steps = max_steps
        self._grad_steps = grad_steps
        self._seed = jax.random.PRNGKey(seed)
        # Priors
        self._prior_z = jnp.full(self._num_classes, fill_value=1 / num_classes)
        self._alpha_prior = alpha_prior or tfd.Normal(loc=0.0, scale=1.0)
        self._log_beta_prior = log_beta_prior or tfd.Normal(loc=0.0, scale=1.0)
        # Initialize optimizer state
        self._optimizer = Optim(*optimizers.momentum(learning_rate, mass=0.9))
        self._optimizer.init_state_with(
            params={
                "alpha": self._alpha_prior.sample(num_labelers, seed=self._seed),
                "log_beta": self._log_beta_prior.sample(num_tasks, seed=self._seed),
            },
        )
        # cls mask is a matrix where each element in first dim is a tensor of the same elements.
        # FIXME: Probably would throw an exception for big datasets
        # However, iterating over batches would solve this problem
        self._cls_mask = repeat(
            jnp.arange(self._num_classes),
            "cls -> cls tasks labelers",
            tasks=num_tasks,
            labelers=num_labelers,
        )
        self._indices = {"alpha": None, "log_beta": None}

    @property
    def alpha(self) -> jnp.ndarray:
        params = self._optimizer.params()
        if params is None:
            raise Exception("First you need to fit GLAD on some data.")
        return params["alpha"]

    @property
    def log_beta(self) -> jnp.ndarray:
        params = self._optimizer.params()
        if params is None:
            raise Exception("First you need to fit GLAD on some data.")
        return params["log_beta"]

    @property
    def inv_beta(self) -> jnp.ndarray:
        params = self._optimizer.params()
        if params is None:
            raise Exception("First you need to fit GLAD on some data.")
        return 1 / jnp.exp(params["log_beta"])

    def fit(
        self,
        data: np.ndarray,
        alpha_idx: np.ndarray = None,
        log_beta_idx: np.ndarray = None,
        prior: List[float] = None,
    ) -> None:
        # data ~ (tasks, labelers)
        self._inner_data = jnp.asarray(data)
        self._prior_z = jnp.array(prior) if prior is not None else self._prior_z
        self._indices["alpha"], self._indices["log_beta"] = alpha_idx, log_beta_idx
        with alive_bar(self._max_steps, title="Whose vote should count more?") as bar:
            prev_loss = 1e13
            for _ in range(self._max_steps):
                # TODO: Если датасет большой, будем тут делать итерацию по данным
                loss = self._m_step(self._e_step(self._optimizer.params()))
                # Condition to stop EM-algorithm.
                if abs(loss - prev_loss) < self._tol:
                    break
                prev_loss = loss
                # Update progress bar
                bar()
                bar.text(f"loss: {loss:.4f}")
        logger.info(f"Final Log-Likelihood: {loss:.4f}")

    def result(self, reset: bool = False) -> None:
        result = self._e_step(self._optimizer.params()).argmax(axis=-1)
        if reset:
            self.reset()
        return result

    def reset(self) -> None:
        self._inner_data = None
        self._optimizer.state = None
        self._prior_z = jnp.full(self._num_classes, fill_value=1 / self._num_classes)
        self._indices = {"alpha": None, "log_beta": None}

    @partial(jax.jit, static_argnums=(0,))
    def _e_step(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        params = {
            param: tensor if self._indices[param] is None else tensor[self._indices[param]]
            for param, tensor in params.items()
        }
        # log_prior ~ (num classes)
        log_pz = jnp.log(self._prior_z)
        # alpha_bets ~ (tasks, labelers)
        alpha_beta = jnp.einsum("l,t->tl", params["alpha"], jnp.exp(params["log_beta"]))
        # equality ~ (num classes, tasks, labelers)
        equality = self._inner_data == self._cls_mask
        # log_pl ~ (num classes, tasks, labelers) -> (tasks, num_classes)
        log_pl = reduce(
            (
                equality * -jax.nn.softplus(-alpha_beta)
                + ~equality * (-jax.nn.softplus(alpha_beta) - math.log(self._num_classes - 1))
            ),
            "cls tasks labelers -> tasks cls",
            reduction="sum",
        )
        return jax.nn.softmax(log_pz + log_pl, axis=-1)

    def _m_step(self, q_z: jnp.ndarray) -> jnp.ndarray:
        # q_z ~ (tasks, num_classes)
        forward_pass = jax.value_and_grad(self._m_step_loss_fn)
        for _ in range(self._grad_steps):
            loss, grad = forward_pass(self._optimizer.params(), q_z=q_z)
            self._optimizer.step(grad)
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def _m_step_loss_fn(self, params: Dict[str, jnp.ndarray], q_z: jnp.ndarray) -> jnp.ndarray:
        params = {
            param: tensor if self._indices[param] is None else tensor[self._indices[param]]
            for param, tensor in params.items()
        }
        # alpha ~ (labelers)
        # log_beta ~ (tasks)
        # q_z ~ (tasks, num_classes)
        # alpha_beta ~ (tasks, labelers)
        alpha_beta = jnp.einsum("l,t->tl", params["alpha"], jnp.exp(params["log_beta"]))
        # equality ~ (num classes, tasks, labelers)
        equality = self._inner_data == self._cls_mask
        # log_pl ~ (num classes, tasks, labelers)
        log_pl = (
            equality * -jax.nn.softplus(-alpha_beta)
            + ~equality * (-jax.nn.softplus(alpha_beta) - math.log(self._num_classes - 1))
        )
        return -jnp.einsum("tc,ctl->", q_z, log_pl)


# MaxVoting requires answers to start from 0
class MaxVoting:
    def __init__(self) -> None:
        self._inner_data = None
        self._cls_mask = None

    def fit(self, data: np.ndarray) -> None:
        self._inner_data = jnp.asarray(data)
        # FIXME: Probably would throw an exception for big datasets.
        # However, iterating over batches would solve this problem
        self._cls_mask = repeat(
            jnp.unique(self._inner_data),
            "cls -> cls tasks labelers",
            tasks=self._inner_data.shape[0],
            labelers=self._inner_data.shape[-1],
        )

    @partial(jax.jit, static_argnums=(0, 1))
    def result(self, reset: bool = False) -> jnp.ndarray:
        # TODO: It could be easily replaced with scatter_add operation.
        # However, numpy/jax doesn't have a proper and fast implementation of this operation
        uniques = jnp.unique(self._inner_data)
        answers = jnp.zeros((self._inner_data.shape[0], len(uniques)))
        for x in uniques:
            answer_x_sums = (self._inner_data == x).sum(axis=-1)
            answers[:, x] = answer_x_sums
        return jnp.argmax(answers, axis=-1)

    @partial(jax.jit, static_argnums=(0, 1))
    def result_v2(self, reset: bool = False) -> jnp.ndarray:
        # uniques ~ (num classes)
        # answers_freq ~ (num classes, tasks, labelers) -> (num classes, tasks)
        answers_freq = (self._inner_data == self._cls_mask).sum(axis=-1)
        return answers_freq.argmax(axis=0)

    def reset(self) -> None:
        self._inner_data = None
