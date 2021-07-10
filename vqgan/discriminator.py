import pickle

import haiku as hk
import jax
import jax.numpy as jnp

from .utils import hk_init_context, make_conv, maybe_hk_dropout, PmeanBatchNormWithoutState, torch_conv_kernel_to_hk

def load_lpips(config, vgg_func, vgg_params, weights_location='weights/lpips.pkl'):
    with open(weights_location, 'rb') as f:
        pretrained_weights = pickle.load(f)

    def _fn(input_, target, vgg_params):
        return LPIPS(pretrained_weights)(input_, target, vgg_params, vgg_func=vgg_func)

    lpips_model = hk.without_apply_rng(hk.transform(_fn))
    dummy_inp = jnp.zeros((3, config.resolution, config.resolution))
    params = lpips_model.init(None, dummy_inp, dummy_inp, vgg_params)
    return lpips_model, params

class LPIPS(hk.Module):
    def __init__(self, init_weights):
        super().__init__('lpips')
        self.init_weights = init_weights

    def __call__(self, input_, target, vgg_params, *, vgg_func=None):
        input_ = jax.image.resize(input_, (3, 224, 224), 'bilinear')
        target = jax.image.resize(target, (3, 224, 224), 'bilinear')

        scaling = ScalingLayer()
        norm_inp = scaling(input_)
        norm_target = scaling(target)

        in_feats = vgg_func(vgg_params, norm_inp, output_features=True, output_logits=False)['features']
        targ_feats = vgg_func(vgg_params, norm_target, output_features=True, output_logits=False)['features']

        loss = 0

        for in_feat, targ_feat, (weight_name, feat_weight) in zip(in_feats,
                                                   targ_feats,
                                                   self.init_weights.items()):

            f_in = normalize_tensor(in_feat)
            f_targ = normalize_tensor(targ_feat)
            diff = (f_in - f_targ) ** 2

            lin = NetLinLayer(torch_conv_kernel_to_hk(feat_weight))

            mapped_diff = lin(diff)
            spatial_average = jnp.mean(mapped_diff, axis=[1, 2])

            loss += spatial_average
        return loss

class ScalingLayer(hk.Module):
    def __call__(self, img):
        mean = jnp.array([-.030, -.088, -.188])[:, None, None]
        std = jnp.array([.458, .448, .450])[:, None, None]
        return (img - mean) / std

class NetLinLayer(hk.Module):
    def __init__(self, feat_weights, name=None):
        super().__init__(name=name)
        self.feat_weights = feat_weights

    def __call__(self, x):
        x = maybe_hk_dropout(0.5, x)
        conv = hk.Conv2D(1,
                         kernel_shape=1,
                         stride=1,
                         padding=(0, 0),
                         with_bias=False,
                         data_format='NCHW',
                         w_init=hk.initializers.Constant(self.feat_weights))

        return conv(x)

class NLayerDiscriminator(hk.Module):
    def __call__(self, x):
        n_layers = 3
        ndf = 64

        kw = 4

        conv = make_conv(ndf, in_channels=x.shape[0], kernel_shape=kw, stride=2, data_format='NCHW')
        x = conv(x)
        x = jax.nn.leaky_relu(x, 0.2)

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult = min(2 ** n, 8)
            conv = make_conv(
                ndf * nf_mult,
                in_channels=x.shape[0],
                kernel_shape=kw,
                stride=2,
                with_bias=False,
                data_format='NCHW'
            )
            norm = PmeanBatchNormWithoutState()

            x = conv(x)
            x = norm(x)
            x = jax.nn.leaky_relu(x, 0.2)

        conv = make_conv(1, in_channels=x.shape[0], kernel_shape=kw, data_format='NCHW')
        return conv(x)

def make_discriminator(config):
    def discriminator(x):
        return NLayerDiscriminator()(x)

    model = hk.transform(discriminator)
    with hk_init_context():
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((3, config.resolution, config.resolution)))
    return model, params

def normalize_tensor(x):
    norm_factor = jnp.sqrt(jnp.sum(x ** 2, axis=0, keepdims=True))
    return x / (norm_factor + 1e-10)
