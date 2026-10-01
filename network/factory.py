"""Head dispatch: cfg.diffusion.type selects the LightningModule class.

'ddpm' (or an absent key) resolves to MotionDiffusionModule.  The other heads register here via local imports so
that importing this module stays cheap and the ddpm path never imports their dependencies.
"""
from network.lit_module import MotionDiffusionModule


def build_module(cfg, meta):
    kind = str(cfg.diffusion.get('type', 'ddpm'))
    # Structured conditions (learned style / persona tokens) exist in the latent prior and in the raw DDPM / FM heads
    # (the paper gives every baseline the same conditions); the vae codec takes no persona.
    structured = str(cfg.model.get('style_cond', 'none')) != 'none' or str(cfg.model.get('persona_cond', 'text')) != 'text'
    if structured and kind not in ('latfm', 'ddpm', 'fm'):
        raise SystemExit(f'model.style_cond / model.persona_cond are implemented for the latfm, ddpm and fm heads only '
                         f'(diffusion.type={kind}); the vae heads take no persona')
    if kind == 'ddpm':
        return MotionDiffusionModule(cfg, meta)
    if kind == 'fm':
        from network.fm import FlowMatchingModule
        return FlowMatchingModule(cfg, meta)
    if kind == 'vae':
        from network.vae import VAEModule
        return VAEModule(cfg, meta)
    if kind == 'vae_inv':
        from network.vae import VAEInvModule
        return VAEInvModule(cfg, meta)
    if kind == 'latfm':
        from network.latfm import LatentFMModule
        return LatentFMModule(cfg, meta)
    raise SystemExit(f'unknown diffusion.type {kind!r} (ddpm | fm | vae | vae_inv | latfm)')
