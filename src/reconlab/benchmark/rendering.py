"""Interchangeable Gaussian renderers with a shared estimated-geometry contract."""

from dataclasses import dataclass


@dataclass
class RenderOutput:
    rgb: object
    alpha: object
    info: dict
    normal: object = None
    surface_normal: object = None
    distortion: object = None


class GaussianRenderer:
    def __init__(self, method, schedule_scale=1.0):
        if method not in {"3dgs", "3dgs_mcmc", "2dgs", "3dgut_mcmc"}:
            raise ValueError(method)
        self.method = method
        self.schedule_scale = schedule_scale

    def render(self, params, viewmats, intrinsics, sh_degree=3):
        import torch
        from gsplat import rasterization, rasterization_2dgs

        args = dict(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=torch.cat([params["sh0"], params["shN"]], 1),
            viewmats=viewmats,
            Ks=intrinsics,
            width=512,
            height=512,
            sh_degree=sh_degree,
            backgrounds=torch.ones((len(viewmats), 3), device=viewmats.device),
            packed=False,
            near_plane=0.01,
            far_plane=100,
        )
        if self.method == "2dgs":
            rgb, alpha, normal, surface, distortion, _, info = rasterization_2dgs(
                **args, render_mode="RGB+ED", distloss=True
            )
            return RenderOutput(rgb[..., :3], alpha, info, normal, surface, distortion)
        if self.method == "3dgut_mcmc":
            args.update(with_ut=True, with_eval3d=True)
        rgb, alpha, info = rasterization(**args)
        return RenderOutput(rgb, alpha, info)

    def strategy(self):
        from gsplat.strategy import DefaultStrategy, MCMCStrategy

        if self.method.endswith("mcmc"):
            strategy = MCMCStrategy(cap_max=200000)
        elif self.method == "2dgs":
            strategy = DefaultStrategy(key_for_gradient="gradient_2dgs")
        else:
            strategy = DefaultStrategy()
        # Follow the pinned gsplat example's adjust_steps for iteration schedules.
        for name in ["refine_start_iter", "refine_stop_iter", "refine_every", "reset_every"]:
            if hasattr(strategy, name):
                setattr(strategy, name, max(1, int(getattr(strategy, name) * self.schedule_scale)))
        if getattr(strategy, "noise_injection_stop_iter", -1) >= 0:
            strategy.noise_injection_stop_iter = max(
                1, int(strategy.noise_injection_stop_iter * self.schedule_scale)
            )
        return strategy

    def regularization(self, params, output, step):
        loss = 0.0
        if self.method.endswith("mcmc"):
            loss = (
                0.01 * params["opacities"].sigmoid().mean() + 0.01 * params["scales"].exp().mean()
            )
        if self.method == "2dgs":
            if step > int(7000 * self.schedule_scale):
                surface = output.surface_normal
                if surface.ndim == 3:
                    surface = surface[None]
                loss = (
                    loss
                    + 0.05 * (1 - (output.normal * surface * output.alpha.detach()).sum(-1)).mean()
                )
            if step > int(3000 * self.schedule_scale):
                loss = loss + 0.01 * output.distortion.mean()
        return loss
