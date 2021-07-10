import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .config import ConfigModule
from .utils import interpolate_nearest, make_conv, maybe_hk_dropout, swish

def group_norm(x):
    return hk.GroupNorm(groups=32, eps=1e-6, data_format='channels_first')(x[None])[0]

class ResnetBlock(ConfigModule):
    def __init__(self, config, out_channels, name=None):
        super().__init__(config, name)
        self.out_channels = out_channels

    def __call__(self, x):
        conv1 = make_conv(self.out_channels, in_channels=x.shape[0])

        conv2 = make_conv(self.out_channels, in_channels=self.out_channels)

        h = group_norm(x)
        h = swish(h)
        h = conv1(h)

        h = group_norm(h)
        h = swish(h)

        h = maybe_hk_dropout(self.config.dropout, h)

        h = conv2(h)

        if x.shape[0] != h.shape[0]:
            x = make_conv(self.out_channels, kernel_shape=1, padding=(0, 0), in_channels=x.shape[0])(x)

        return x + h

class AttnBlock(hk.Module):
    def __call__(self, x):

        in_ch = x.shape[0]

        proj_q = make_conv(in_ch, in_channels=in_ch, padding=(0, 0), kernel_shape=1)
        proj_k = make_conv(in_ch, in_channels=in_ch, padding=(0, 0), kernel_shape=1)
        proj_v = make_conv(in_ch, in_channels=in_ch, padding=(0, 0), kernel_shape=1)
        proj_out = make_conv(in_ch, in_channels=in_ch, padding=(0, 0), kernel_shape=1)

        h = group_norm(x)

        q = proj_q(h)
        k = proj_k(h)
        v = proj_v(h)

        _, height, width = q.shape

        seq_len = height * width

        q = q.reshape(-1, seq_len)
        k = k.reshape(-1, seq_len)

        attn_logits = jnp.einsum('fs,ft->st', q, k)
        attn_logits /= np.sqrt(in_ch)

        attn_weights = jax.nn.softmax(attn_logits)

        v = v.reshape(-1, seq_len)
        h = jnp.einsum('fs,ts->ft', v, attn_weights)
        h = h.reshape((in_ch, height, width))

        h = proj_out(h)
        return x + h

class Downsample(hk.Module):
    def __call__(self, x):
        conv = make_conv(x.shape[0], in_channels=x.shape[0], stride=2, padding=(0, 0))

        x = jnp.pad(x, ((0, 0), (0, 1), (0, 1)))
        return conv(x)

class Upsample(hk.Module):
    def __call__(self, x):
        conv = make_conv(x.shape[0], in_channels=x.shape[0])

        x = interpolate_nearest(x, 2.0)
        return conv(x)

class Encoder(ConfigModule):
    def __call__(self, x):
        conv_in = make_conv(self.config.channels, in_channels=x.shape[0])

        num_resolutions = len(self.config.ch_mult)

        curr_res = self.config.resolution

        h = conv_in(x)
        for i_level in range(num_resolutions):
            block_out = self.config.channels * self.config.ch_mult[i_level]
            for _ in range(self.config.num_res_blocks):
                res_block = ResnetBlock(
                    self.config,
                    out_channels=block_out,
                )

                h = res_block(h)
                if curr_res in self.config.attn_resolutions:
                    attn = AttnBlock()
                    h = attn(h)

            if i_level < num_resolutions - self.config.no_downscale_layers:
                down = Downsample()
                h = down(h)
                curr_res //= 2

        mid_block_1 = ResnetBlock(self.config, block_out)
        mid_attn = AttnBlock()

        mid_block_2 = ResnetBlock(self.config, block_out)

        h = mid_block_1(h)
        h = mid_attn(h)
        h = mid_block_2(h)

        h = group_norm(h)
        h = swish(h)

        conv_out = make_conv(self.config.z_channels, in_channels=h.shape[0])
        h = conv_out(h)

        return h

class Decoder(ConfigModule):
    def __call__(self, z):
        num_resolutions = len(self.config.ch_mult)
        curr_res = z.shape[-1]

        block_in = self.config.channels * self.config.ch_mult[-1]

        conv_in = make_conv(block_in, in_channels=z.shape[0])

        h = conv_in(z)

        mid_block_1 = ResnetBlock(self.config, block_in)
        mid_attn = AttnBlock()
        mid_block_2 = ResnetBlock(self.config, block_in)

        h = mid_block_1(h)
        h = mid_attn(h)
        h = mid_block_2(h)

        for i_level in reversed(range(num_resolutions)):
            block_out = self.config.channels * self.config.ch_mult[i_level]
            for _ in range(self.config.num_res_blocks + 1):
                block = ResnetBlock(self.config, block_out)
                h = block(h)
                if curr_res in self.config.attn_resolutions:
                    h = AttnBlock()(h)
            if i_level >= self.config.no_downscale_layers:
                h = Upsample()(h)
                curr_res *= 2


        conv_out = make_conv(self.config.out_channels, in_channels=h.shape[0], name='final_conv')

        h = group_norm(h)
        h = swish(h)
        h = conv_out(h)

        return h

def quantize(z_flat, embedding):
    emb_weights = embedding.embeddings

    z_sq_norm = jnp.sum(z_flat ** 2, axis=0)
    emb_sq_norm = jnp.sum(emb_weights ** 2, axis=1)

    cross_prods = jnp.einsum('ct,vc->tv', z_flat, emb_weights)

    sq_dists = z_sq_norm[:, None] + emb_sq_norm[None, :] - 2 * cross_prods

    best_indices = jnp.argmin(sq_dists, axis=1)
    selections = jnp.zeros((best_indices.shape[0], emb_weights.shape[0]))
    selections = selections.at[jnp.arange(best_indices.shape[0]), best_indices].set(1)

    z_q = embedding(best_indices)
    return z_q, best_indices, selections

class VectorQuantizer(ConfigModule):
    def __call__(self, z, embeddings):
        z_flat = z.reshape(z.shape[0], -1)
        z_q_flat, best_indices, selections = quantize(z_flat, embeddings)

        z_q = z_q_flat.reshape(z.shape)

        loss = jnp.mean((jax.lax.stop_gradient(z_q) - z) ** 2)
        loss += self.config.beta * jnp.mean((z_q - jax.lax.stop_gradient(z)) ** 2)

        z_q = z + jax.lax.stop_gradient(z_q - z)

        usage_mean = jnp.mean(selections, axis=0)
        ppl = jnp.exp(-jnp.sum(usage_mean * jnp.log(usage_mean + 1e-10)))

        return z_q, loss, (ppl, selections, best_indices)

class VQGanModel(ConfigModule):
    def __call__(self, input_=None, codes=None):
        assert input_ is not None or codes is not None
        result = {}
        embeddings = hk.Embed(
            self.config.n_embed,
            self.config.embed_dim,
            # w_init=hk.initializers.RandomUniform(-1 / self.config.n_embed, 1 / self.config.n_embed)
            w_init=hk.initializers.RandomUniform(-self.config.emb_init_scale, self.config.emb_init_scale)
        )
        if input_ is not None:
            encoder = Encoder(self.config)
            quantizer = VectorQuantizer(self.config)

            h = encoder(input_)

            quant_conv = make_conv(
                self.config.embed_dim,
                in_channels=h.shape[0],
                kernel_shape=1,
                padding=(0, 0),
                name='quant_conv'
            )
            h = quant_conv(h)

            quant, emb_loss, (ppl, selections, best_indices) = quantizer(h, embeddings)
            result['encoded'] = h
            result['selected_codes'] = best_indices
            result['embedding_loss'] = emb_loss
        else:
            quant = embeddings(codes)
            quant = jnp.transpose(quant, (2, 0, 1))

        post_quant_conv = make_conv(
            self.config.z_channels,
            in_channels=quant.shape[1],
            kernel_shape=1,
            padding=(0, 0),
            name='post_quant_conv'
        )
        decoder = Decoder(self.config)

        quant = post_quant_conv(quant)
        dec = decoder(quant)
        result['decoded'] = dec
        return result

def make_vqgan(config):
    def vqgan(x):
        return VQGanModel(config)(x)

    model = hk.transform(vqgan)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((3, config.resolution, config.resolution)))
    return model, params
