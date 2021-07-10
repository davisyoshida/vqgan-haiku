from collections import Counter
from functools import partial


import haiku as hk
import jax
from jax.experimental.optimizers import adam
import jax.numpy as jnp

from vgg.vgg import get_model

from .discriminator import load_lpips, make_discriminator
from .vqgan import make_vqgan, VQGanModel
from .utils import compose


def build(config):
    vgg_model, vgg_params = get_model(weights_location='weights/vgg.pkl')
    vgg_func = vgg_model.apply

    lpips_model, lpips_params = load_lpips(config, vgg_func, vgg_params, weights_location='weights/lpips.pkl')
    lpips_func = lpips_model.apply

    vqgan_model, vqgan_params = make_vqgan(config)
    vqgan_func = vqgan_model.apply

    disc_model, disc_params = make_discriminator(config)
    disc_func = disc_model.apply

    if config.warmup_steps is None:
        step_size = config.learning_rate
    else:
        def step_size(t):
            return jnp.minimum(1., t / config.warmup_steps) * config.learning_rate

    opt_init, opt_update, get_params = adam(step_size=step_size, b1=0.5, b2=0.9)

    vqgan_opt_state = opt_init(vqgan_params)
    disc_opt_state = opt_init(disc_params)

    batch_lpips = jax.vmap(lpips_func, (None, 0, 0, None))
    batch_vqgan = jax.vmap(vqgan_func, (None, 0, 0))
    batch_disc = jax.vmap(disc_func, (None, 0, 0), axis_name='batch')

    def compute_rec_loss(lpips_params, vgg_params, inputs, reconstructions):
        l1_loss = jnp.mean(jnp.abs(inputs - reconstructions))
        percept_loss = jnp.mean(batch_lpips(lpips_params, inputs, reconstructions, vgg_params))
        return config.l1_weight * l1_loss + config.percep_weight * percept_loss, (l1_loss, percept_loss)

    rec_loss_grad_fn = jax.value_and_grad(compute_rec_loss, argnums=3, has_aux=True)

    def disc_loss(disc_params, disc_key, im, reconstruction_logits):
        logits_real = disc_func(disc_params, disc_key, im)
        loss_real = jnp.mean(jax.nn.relu(1 - logits_real))
        loss_fake = jnp.mean(jax.nn.relu(1 + reconstruction_logits))
        return 0.5 * (loss_real + loss_fake)

    batch_disc_loss = jax.vmap(disc_loss, (None, 0, 0, 0), axis_name='batch')
    disc_loss_grad = jax.value_and_grad(
        compose(jnp.mean, batch_disc_loss),
        argnums=(0, 3))

    def grads_and_losses(vqgan_params, disc_params, lpips_params, vgg_params, inputs, t, run_discriminator):
        dropout_key = jax.random.PRNGKey(t)

        disc_key, disc_loss_key, *batch_vqgan_keys = jax.random.split(dropout_key, 2 + inputs.shape[0])
        batch_disc_keys = jax.random.split(disc_key, inputs.shape[0])
        batch_disc_loss_keys = jax.random.split(disc_loss_key, inputs.shape[0])

        batch_vqgan_keys = jnp.stack(batch_vqgan_keys)
        batch_disc_keys = jnp.stack(batch_disc_keys)
        batch_disc_loss_keys = jnp.stack(batch_disc_loss_keys)

        def reconstruct(vqgan_params):
            model_output = batch_vqgan(vqgan_params, batch_vqgan_keys, inputs)
            reconstructions = model_output['decoded']
            codebook_loss = model_output['embedding_loss']
            return reconstructions, jnp.mean(codebook_loss)

        (reconstructions, codebook_loss), gen_vjp = jax.vjp(reconstruct, vqgan_params)

        (_, (l1_loss, percept_loss)), rec_grad = rec_loss_grad_fn(lpips_params, vgg_params, inputs, reconstructions)

        codebook_pullback_input = jnp.ones_like(codebook_loss) / codebook_loss.size * config.codebook_weight

        reconstruction_pullback_input = rec_grad

        disc_full_grad = None
        disc_loss = 0.
        gen_disc_loss = 0.
        d_weight = 0
        if run_discriminator:
            # Hopefully the JIT kills the extra computation this requires
            nll_grad_for_last_layer, = gen_vjp((rec_grad, 0.))
            last_layer_key = 'vq_gan_model/decoder/final_conv'
            last_layer_nll_grad = nll_grad_for_last_layer[last_layer_key]['w']

            rec_logits, disc_vjp_func = jax.vjp(batch_disc, disc_params, batch_disc_keys, reconstructions)

            # Get the gradient of the discriminator output w.r.t. the reconstructions
            gen_disc_loss = -jnp.mean(rec_logits)
            _, _, rec_gen_grad = disc_vjp_func(-jnp.ones_like(rec_logits) / rec_logits.size)

            # Push grad back through VQGAN vjp
            # Again hopefully the JIT gets rid of the unused computations
            gen_disc_grad_for_last_layer, = gen_vjp((rec_gen_grad, 0.))

            last_layer_gen_grad = gen_disc_grad_for_last_layer[last_layer_key]['w']

            nll_grad_norm = jnp.linalg.norm(last_layer_nll_grad)
            gen_grad_norm = jnp.linalg.norm(last_layer_gen_grad)

            d_weight = nll_grad_norm / (gen_grad_norm + 1e-4)
            d_weight = jnp.minimum(d_weight, 1e4) * config.disc_weight

            # Directly compute discriminator gradient for real images, and get grad w.r.t.
            # reconstructions for use with previously computed gen_vjp
            disc_loss, (disc_real_grad, disc_rec_logit_grad) = disc_loss_grad(disc_params,
                                                                              batch_disc_loss_keys,
                                                                              inputs,
                                                                              rec_logits)
            disc_fake_grad, _, _ = disc_vjp_func(disc_rec_logit_grad)

            disc_full_grad = jax.tree_multimap(jnp.add, disc_real_grad, disc_fake_grad)

            reconstruction_pullback_input += d_weight * rec_gen_grad

        vqgan_grad, = gen_vjp((reconstruction_pullback_input, codebook_pullback_input))

        return {
            'grads': {
                'vqgan': vqgan_grad,
                'disc': disc_full_grad
            },
            'losses': {
                'l1': l1_loss,
                'perceptual': percept_loss,
                'disc': disc_loss,
                'gen_disc': gen_disc_loss,
                'codebook_loss': codebook_loss
            },
        }

    @partial(jax.jit, donate_argnums=(2, 3), static_argnums=(6,))
    def _step(vgg_params, lpips_params, vqgan_opt_state, disc_opt_state, inputs, t, run_discriminator=False):
        results = grads_and_losses(vqgan_params=get_params(vqgan_opt_state),
                                   disc_params=get_params(disc_opt_state),
                                   lpips_params=lpips_params,
                                   vgg_params=vgg_params,
                                   inputs=inputs,
                                   t=t,
                                   run_discriminator=run_discriminator)

        grads = results['grads']
        vqgan_opt_state = opt_update(t, grads['vqgan'], vqgan_opt_state)
        if run_discriminator:
            disc_opt_state = opt_update(t - config.disc_start_step, grads['disc'], disc_opt_state)

        return vqgan_opt_state, disc_opt_state, results['losses']

    _step = partial(_step, vgg_params, lpips_params)

    # doing this so you can use kwargs to call the JITted function
    def step(vqgan_opt_state, disc_opt_state, inputs, t, run_discriminator):
        return _step(vqgan_opt_state, disc_opt_state, inputs, t, run_discriminator)

    return {
        'step_fn': step,
        'model': vqgan_model,
        'init_opt_state': vqgan_opt_state,
        'init_disc_opt_state': disc_opt_state,
        'get_params': get_params # use get_params(opt_state) if you need to access the parameters directly
    }


def build_decoder(config):
    def vqgan(z):
        return VQGanModel(config)(codes=z)

    model = hk.transform(vqgan)
    return model

