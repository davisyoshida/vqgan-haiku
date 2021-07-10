from collections import namedtuple

import haiku as hk

VQGanConfig = namedtuple('VQGanConfig', [
    'learning_rate',
    'resolution',
    'no_downscale_layers',
    'embed_dim',
    'n_embed',
    'ch_mult',
    'num_res_blocks',
    'channels',
    'temb_channels',
    'dropout',
    'z_channels',
    'out_channels',
    'attn_resolutions',
    'beta',
    'disc_weight',
    'disc_start_step',
    'codebook_weight',
    'l1_weight',
    'percep_weight',
    'emb_init_scale',
    'warmup_steps'
])

DEFAULT_CONFIG = VQGanConfig(
    learning_rate=4.5e-6,
    resolution=128,
    no_downscale_layers=2,
    embed_dim=256,
    n_embed=1024,
    ch_mult=(1, 1, 2, 2, 4),
    channels=128,
    num_res_blocks=2,
    attn_resolutions=[16],
    temb_channels=-1,
    dropout=0.,
    z_channels=256,
    out_channels=3,
    beta=0.25,
    disc_weight=0.25,
    disc_start_step=10000,
    codebook_weight=1.,
    l1_weight=1.,
    percep_weight=1.,
    emb_init_scale=0.75,
    warmup_steps=1000,
)

class ConfigModule(hk.Module):
    def __init__(self, config : VQGanConfig, name=None):
        super().__init__(name)
        self.config = config
