"""Model zoo: the proposed clocked model, its 2x2 ablation, and baselines.

Every model is a spec dict (see SPECS) plus a torch module built by build_model(spec, ...).
"""

import torch
from torch import nn

from .ssm import SSMBackbone, TransformerBackbone
from .stem import BilinearStem, ExactStem, stem_features


def _spec(name, family, stem=None, feats=("point",), backbone="ssm", clock_r=None, role=""):
    return {"name": name, "family": family, "stem": stem, "feats": tuple(feats), "backbone": backbone,
            "clock_r": clock_r, "role": role}


SPECS = {s["name"]: s for s in [
    # ---- native-rate clock (Delta_c = native dt): clean 2x2 ablation; P1 == A1 and A2 == A3 on the native grid
    _spec("P1_foh_clock", "clock", "foh", ("stem", "point"), clock_r=1, role="proposed: exact stem + fixed clock"),
    _spec("A1_foh_perobs", "perobs", "foh", ("stem", "point"), role="ablation: exact stem only"),
    _spec("A2_zoh_clock", "clock", "zoh", ("stem", "point"), clock_r=1, role="ablation: fixed clock only"),
    _spec("A3_zoh_perobs", "perobs", "zoh", ("stem", "point"), role="ablation: neither"),
    _spec("B_point_clock1", "clock", None, ("point",), clock_r=1, role="baseline: fixed-grid resampling + SSM"),
    _spec("B_dtonly", "perobs", None, ("point",), role="baseline: dt-only step-size rescaling (S5/FlowState-style)"),
    _spec("B_tf_clock1", "clock", None, ("point",), backbone="transformer", clock_r=1, role="baseline: resampling + Transformer"),
    # ---- coarse clock (Delta_c = 4 native dt)
    _spec("P4_foh_clock", "clock", "foh", ("stem", "point"), clock_r=4, role="proposed, coarse clock"),
    _spec("A2_zoh_clock4", "clock", "zoh", ("stem", "point"), clock_r=4, role="ablation: fixed clock only, coarse"),
    _spec("B_point_clock4", "clock", None, ("point",), clock_r=4, role="baseline: fixed-grid resampling + SSM, coarse"),
    _spec("B_binmean_clock4", "clock", None, ("binmean", "point"), clock_r=4, role="baseline: resampling with box prefilter"),
    _spec("B_patch_clock4", "clock", None, ("patch",), clock_r=4, role="baseline: fine resampling + linear patch + SSM"),
    _spec("B_tfpatch_clock4", "clock", None, ("patch",), backbone="transformer", clock_r=4, role="baseline: fine resampling + patch Transformer"),
    _spec("B_nrde_clock4", "nrde", None, (), clock_r=4, role="close work: NRDE-style log-ODE (depth-2 log-signature windows)"),
    _spec("B_rformer_clock4", "clock", None, ("logsig_local", "logsig_global"), backbone="transformer", clock_r=4,
          role="close work: RFormer-style multi-view depth-2 log-signatures + Transformer"),
    _spec("B_bilin_clock4", "clock", "bilin", ("stem", "point"), clock_r=4,
          role="close work: multirate-SSM-style bilinear linear stem + decimation + same SSM backbone"),
]}


def feat_dim(spec, c_in, n_stem):
    f = 0
    for k in spec["feats"]:
        if k == "stem":
            f += 2 * n_stem
        elif k in ("point", "binmean"):
            f += c_in
        elif k == "patch":
            f += spec["clock_r"] * c_in
        elif k in ("logsig_local", "logsig_global"):
            f += (1 + c_in) + (1 + c_in) * c_in // 2
    return f


class NRDE(nn.Module):
    """h_0 = W x(t_start);  h_j = h_{j-1} + f(h_{j-1}) @ logsig_j  (one Euler step of the log-ODE per window).

    logsig features are scaled by their train RMS and divided by sqrt(dim) so that one Euler step
    stays O(1) (pilot fix: without it the state diverged); LayerNorm before the linear head.
    """

    def __init__(self, c_in, ls_dim, H, hidden, n_out, ls_scale):
        super().__init__()
        self.init = nn.Linear(c_in, H)
        self.f = nn.Sequential(nn.Linear(H, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, H * ls_dim), nn.Tanh())
        self.head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, n_out))
        self.register_buffer("ls_scale", torch.as_tensor(ls_scale, dtype=torch.float32))
        self.H, self.D = H, ls_dim

    def forward(self, x0, ls):
        ls = ls / (self.ls_scale * self.D ** 0.5)
        h = self.init(x0)
        for j in range(ls.shape[1]):
            h = h + (self.f(h).view(-1, self.H, self.D) @ ls[:, j, :, None]).squeeze(-1)
        return self.head(h)


class Net(nn.Module):
    def __init__(self, spec, c_in, n_out, cfg, seed, native_dt, ls_scale=None):
        super().__init__()
        self.spec = spec
        torch.manual_seed(seed)
        if spec["family"] == "nrde":
            self.core = NRDE(c_in, 1 + c_in + (1 + c_in) * c_in // 2, cfg["nrde_H"], cfg["nrde_hidden"], n_out, ls_scale)
            return
        self.stem = None
        if ls_scale is not None and any(f.startswith("logsig") for f in spec["feats"]):
            self.register_buffer("feat_scale", torch.as_tensor(ls_scale, dtype=torch.float32))
        if spec["stem"]:
            self.stem = (BilinearStem if spec["stem"] == "bilin" else ExactStem)(c_in, cfg["n_stem"], cfg["stem_tau_min"], cfg["stem_tau_max"], cfg["stem_f_max"], seed=seed)
        f_in = feat_dim(spec, c_in, cfg["n_stem"])
        self.dt_ref = native_dt * (spec["clock_r"] or 1)
        if spec["backbone"] == "ssm":
            self.core = SSMBackbone(f_in, cfg["H"], cfg["N"], cfg["n_layers"], self.dt_ref, cfg["dropout"], n_out, seed=seed)
        else:
            self.core = TransformerBackbone(f_in, cfg["H"], cfg["tf_layers"], cfg["tf_heads"], cfg["dropout"], n_out,
                                            time_scale=self.dt_ref)

    def forward(self, batch, force_scan=False):
        if self.spec["family"] == "nrde":
            return self.core(batch["x0"], batch["ls"])
        feats = batch["feats"]
        if hasattr(self, "feat_scale"):
            feats = feats / self.feat_scale
        if self.stem is not None:
            h = self.stem(batch["d"], batch["xs"], batch["xe"], batch["win"], batch["rem"], batch["qdt"])
            feats = torch.cat([stem_features(h).to(feats.dtype), feats], dim=-1)
        if self.spec["backbone"] == "transformer":
            return self.core(feats, batch["q"], batch["w"])
        if self.spec["family"] == "clock":
            dt = self.dt_ref                                  # fixed physical clock step (python float -> LTI path)
        else:
            dt = batch["dt"]
            if batch["uniform"] and not force_scan:
                dt = float(batch["dt"][0, 0])
        return self.core(feats, dt, batch["w"])
