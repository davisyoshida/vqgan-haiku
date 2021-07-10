"""Various things copied from my personal utils repo"""
from contextlib import contextmanager
from contextvars import ContextVar

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def compose(*funcs):
    def composed_func(*args, **kwargs):
        result = None
        for func in reversed(funcs):
            result = func(*args, **kwargs) if result is None else func(result)
        return result
    return composed_func

def scaled_global_norm(tree):
    size, norm = jax.tree_util.tree_reduce(lambda carry, leaf: (carry[0] + leaf.size, carry[1] + jnp.sum(leaf ** 2)), tree, jnp.zeros(2))
    return (norm / size) ** 0.5

def interpolate_nearest(arr, scale_factor):
    # only supports CWH
    if len(arr.shape) != 3:
        raise ValueError(f'Expected 3 dimensional tensor, got shape: {arr.shape}')

    channels, in_height, in_width = arr.shape

    height = int(in_height * scale_factor)
    width = int(in_width * scale_factor)

    read_channels = jnp.arange(channels)[:, None, None]
    read_rows = jnp.linspace(0, in_height - 1, height)[None, :, None]
    read_cols = jnp.linspace(0, in_width - 1, width)[None, None, :]

    read_rows = jnp.round(read_rows).astype(jnp.int32)
    read_cols = jnp.round(read_cols).astype(jnp.int32)

    return arr[
        jnp.tile(read_channels, (1, height, width)),
        jnp.tile(read_rows, (channels, 1, width)),
        jnp.tile(read_cols, (channels, height, 1)),
    ]

def maybe_hk_dropout(rate, value):
    key = hk.maybe_next_rng_key()
    if key is not None:
        value = hk.dropout(key, rate, value)
    return value

def swish(x):
    return x * jax.nn.sigmoid(x)

def torch_init_conv(n_in, kernel_size, n_spatial=2):
    stddev = 1 / np.sqrt(n_in * (kernel_size ** n_spatial))
    return hk.initializers.RandomUniform(-stddev, stddev)

def make_conv(*args, in_channels, **kwargs):
    kwargs = {
        'kernel_shape': 3,
        'stride': 1,
        'padding': (1, 1),
        'data_format':'NCHW',
        'with_bias': True,
        **kwargs
    }

    init_fn = torch_init_conv(in_channels, kwargs['kernel_shape'])
    kwargs['w_init'] = init_fn
    if kwargs['with_bias']:
        kwargs['b_init'] = init_fn

    return hk.Conv2D(*args, **kwargs)

_in_init = ContextVar('hk_init', default=False)
@contextmanager
def hk_init_context():
    token = _in_init.set(True)
    try:
        yield
    finally:
        _in_init.reset(token)

class PmeanBatchNormWithoutState(hk.Module):
    def __init__(self, axis_name='batch', name='bn'):
        super().__init__(name=name)
        self.axis_name = axis_name

    def __call__(self, x):
        offset = hk.get_parameter('bn_offset', shape=(x.shape[0],), init=hk.initializers.Constant(0.))
        scale = hk.get_parameter('bn_scale', shape=(x.shape[0],), init=hk.initializers.Constant(1.))
        if _in_init.get():
            mean = jnp.zeros(x.shape[0])
            variance = jnp.zeros(x.shape[0])
        else:
            mean = jnp.mean(jax.lax.pmean(x, self.axis_name), axis=[1, 2])
            second_moment = jnp.mean(jax.lax.pmean(x ** 2, self.axis_name), axis=[1,2])

            variance = second_moment - mean ** 2
        mean = mean[:, None, None]
        variance = variance[:, None, None]

        normalized = ((x - mean) / jnp.sqrt(variance + 1e-5)) * scale[:, None, None] + offset[:, None, None]
        return normalized

def torch_conv_kernel_to_hk(kernel):
    return jnp.transpose(kernel, [2, 3, 1, 0])
