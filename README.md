# VQGAN-Haiku
This is a partial port of the [official VQGAN repo](https://github.com/CompVis/taming-transformers/) to JAX with Haiku.

## Setup
Install dependencies:
```sh
pip install -r requirements.txt
```

Download pretrained VGG/LPIPS weights for perceptual loss [here](https://drive.google.com/file/d/1jW_G4SsU9g6zgjv8as2r4JFWgCLHpirJ/view?usp=sharing), and un-tar them in the root directory:
```sh
tar -xzvf weights.tar.gz
```

## Usage

```python
import haiku as hk
import jax
import jax.numpy as jnp

from vqgan.build import build, build_decoder
from vqgan.config import DEFAULT_CONFIG


# This builds the model function, JITs a training step function, and initializes the optimizers
built_model = build(DEFAULT_CONFIG)

vqgan_opt_state = built_model['init_opt_state']
disc_opt_state = built_model['init_disc_opt_state']
train_step = built_model['step_fn']
model = built_model['model']
get_params = built_model['get_params']

# make sure not to keep references to the initial optimizer states so they don't hog
del built_model

# make some fake data
# Images should be VGG norm
batch_size = 2
channels = 3
width = 128
height = 128
input_images = jnp.zeros((batch_size, channels, height, width))

# Train on fake data. Make sure to pass in t so the learning rate schedules work right
for t in range(1, 10):
    vqgan_opt_state, disc_opt_state, losses = train_step(
        vqgan_opt_state=vqgan_opt_state,
        disc_opt_state=disc_opt_state,
        inputs=input_images,
        t=t,
        run_discriminator=t >= DEFAULT_CONFIG.disc_start_step
    )

# get the params out of the optimizer_state
params = get_params(vqgan_opt_state)

# encode and decode an image
deterministic_model = hk.without_apply_rng(model)
model_output = deterministic_model.apply(params, input_images[0])

reconstructed_image = model_output['decoded']
z = model_output['selected_codes']

decoder_only_function = hk.without_apply_rng(build_decoder(DEFAULT_CONFIG))
# decode an image from a latent code
image_from_code = decoder_only_function.apply(params, z.reshape(16, 16))['decoded']
```
