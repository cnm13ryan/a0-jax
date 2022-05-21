"""
AlphaZero training script

Train agent by self-play only.
"""

import random
from functools import partial

import chex
import click
import fire
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import opax
import pax

from alphazero import recurrent_fn, replicate, reset_env, step
from connect_two_game import Connect2Game
from env import Enviroment
from policy_net import PolicyValueNet


@chex.dataclass(frozen=True)
class TrainingExample:
    """AlphaZero training example.

    state: the current state of the game.
    action_weights: the target action weights from MCTS policy.
    value: the target value from self-play result.
    """

    state: chex.Array
    action_weights: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class MoveOutput:
    """The output of a single self-play move.

    state: the current state of game.
    reward: the reward after action the action from MCTS policy.
    terminated: the current state is a terminated state (bad state).
    action_weights: the action weights from MCTS policy.
    """

    state: chex.Array
    reward: chex.Array
    terminated: chex.Array
    action_weights: chex.Array


@partial(jax.jit, static_argnums=(3,))
def collect_batched_selfplay_data(
    agent, env: Enviroment, rng_key: chex.Array, batch_size: int
):
    """Collect a batch of selfplay data using mcts."""

    def single_move(prev, inputs):
        """Execute one self-play move using MCTS.

        This function is designed to be compatible with jax.scan.
        """
        env, rng_key = prev
        del inputs
        rng_key, rng_key_next = jax.random.split(rng_key, 2)
        state = env.canonical_observation()
        terminated = env.is_terminated()
        prior_logits, value = jax.vmap(lambda a, s: a(s), in_axes=(None, 0))(
            agent, state
        )
        root = mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=env)
        policy_output = mctx.gumbel_muzero_policy(
            params=agent,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=16,
            gumbel_scale=1.0,
            max_depth=4,
            invalid_actions=env.board != 0,
        )
        action_weights = policy_output.action_weights
        env, reward = jax.vmap(step)(env, policy_output.action)
        return (env, rng_key_next), MoveOutput(
            state=state,
            action_weights=action_weights,
            reward=reward,
            terminated=terminated,
        )

    env = reset_env(env)
    env = replicate(env, batch_size)
    _, selfplay_data = pax.scan(
        single_move, (env, rng_key), None, length=4, unroll=4, time_major=False
    )
    return selfplay_data


def collect_selfplay_data(
    agent, env, rng_key: chex.Array, batch_size: int, data_size: int
):
    """Collect selfplay data for training."""
    N = data_size // batch_size
    rng_keys = jax.random.split(rng_key, N)
    data = []

    with click.progressbar(rng_keys, label="  self play  ") as bar:
        for rng_key in bar:
            batch = collect_batched_selfplay_data(agent, env, rng_key, batch_size)
            data.append(jax.device_get(batch))
    data = jax.tree_map(lambda *xs: np.concatenate(xs), *data)
    return data


def prepare_training_data(data: MoveOutput):
    """Preprocess the data collected from selfplay.

    1. remove states after the enviroment is terminated.
    2. compute the value at each state
    """
    buffer = []
    N = len(data.terminated)
    for i in range(N):
        state = data.state[i]
        is_terminated = data.terminated[i]
        action_weights = data.action_weights[i]
        reward = data.reward[i]
        L = len(is_terminated)
        value = 0.0
        for j in range(L):
            idx = L - 1 - j
            if is_terminated[idx]:
                continue
            if reward[idx] != 0:
                value = reward[idx]
            else:
                value = -value
            buffer.append(
                TrainingExample(
                    state=state[idx],
                    action_weights=action_weights[idx],
                    value=np.array(value, dtype=np.float32),
                )
            )

    return buffer


def loss_fn(net, data: TrainingExample):
    action_logits, value = net(data.state)
    mse_loss = jnp.mean(jnp.square(value - data.value))
    action_logits = jax.nn.log_softmax(action_logits)
    cross_entropy_loss = jnp.einsum("NA,NA->N", data.action_weights, action_logits)
    cross_entropy_loss = -jnp.mean(cross_entropy_loss)
    return mse_loss + cross_entropy_loss, (mse_loss, cross_entropy_loss)


@jax.jit
def train_step(net, optim, data: TrainingExample):
    (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(net, data)
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, losses


def play_against_agent(agent, env):
    env = reset_env(env)
    env.render()
    for i in range(4):
        if i % 2 == 1:
            print("Observation", env.canonical_observation())
            logits, value = agent(env.canonical_observation())
            print(logits, value)
            action = jnp.argmax(logits, axis=-1).item()
            env, reward = step(env, action)
            print(f"agent select action {action} get reward {reward}")
        else:
            action = int(input())
            env, reward = step(env, action)
            print(f"human select action {action} get reward {reward}")
        env.render()
        if env.is_terminated().item():
            break


def train(batch_size: int = 32, num_iterations: int = 500, learing_rate: float = 0.001):
    agent = PolicyValueNet()
    env = Connect2Game()
    rng_key = jax.random.PRNGKey(42)

    optim = opax.chain(
        opax.trace(0.9),
        opax.add_decayed_weights(1e-4),
        opax.scale(learing_rate),
    ).init(agent.parameters())

    for iteration in range(num_iterations):
        print(f"Iteration {iteration}")
        rng_key_1, rng_key = jax.random.split(rng_key, 2)
        data = collect_selfplay_data(agent, env, rng_key_1, batch_size, 102400)
        buffer = prepare_training_data(data)
        random.shuffle(buffer)
        buffer = jax.tree_map(lambda *xs: np.stack(xs), *buffer)
        N = buffer.state.shape[0]
        losses = []
        with click.progressbar(
            range(0, N - batch_size, batch_size), label="  train agent"
        ) as bar:
            for i in bar:
                batch = jax.tree_map(lambda x: x[i : (i + batch_size)], buffer)
                agent, optim, loss = train_step(agent, optim, batch)
                losses.append(loss)

        mse_loss, cross_entropy_loss = zip(*losses)
        mse_loss = sum(mse_loss).item() / len(mse_loss)
        cross_entropy_loss = sum(cross_entropy_loss).item() / len(cross_entropy_loss)
        print(
            f"  train losses:  mse {mse_loss:.3f}  cross entropy {cross_entropy_loss:.3f}"
        )

    play_against_agent(agent, env)


if __name__ == "__main__":
    fire.Fire(train)
