from __future__ import annotations

import torch


@torch.inference_mode()
def ddim_sample(
    model,
    audio: torch.Tensor,
    controls: torch.Tensor,
    alpha_cumulative: torch.Tensor,
    *,
    latent_channels: int = 16,
    latent_length: int = 384,
    sampling_steps: int = 50,
    control_guidance: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch = audio.shape[0]
    latent = torch.randn(
        (batch, latent_channels, latent_length),
        device=audio.device,
        dtype=audio.dtype,
        generator=generator,
    )
    timesteps = torch.linspace(
        alpha_cumulative.shape[0] - 1,
        0,
        sampling_steps,
        device=audio.device,
    ).round().long().unique_consecutive()
    for index, step in enumerate(timesteps):
        timestep = torch.full((batch,), int(step), device=audio.device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.float16):
            conditional = model(latent, timestep, audio, controls)
            if control_guidance != 1.0:
                drop_mask = torch.ones(batch, device=audio.device, dtype=torch.bool)
                unconditional = model(
                    latent,
                    timestep,
                    audio,
                    controls,
                    control_drop_mask=drop_mask,
                )
                predicted_noise = unconditional + control_guidance * (conditional - unconditional)
            else:
                predicted_noise = conditional

        alpha = alpha_cumulative[step].to(latent.dtype)
        predicted_clean = (latent - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
        predicted_clean = predicted_clean.clamp(-8.0, 8.0)
        if index + 1 == len(timesteps):
            latent = predicted_clean
            break
        next_alpha = alpha_cumulative[timesteps[index + 1]].to(latent.dtype)
        latent = next_alpha.sqrt() * predicted_clean + (1.0 - next_alpha).sqrt() * predicted_noise
    return latent
