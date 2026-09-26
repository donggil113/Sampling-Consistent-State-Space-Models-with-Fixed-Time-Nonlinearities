"""Numerical checks for src/fxclock (float64 unless stated).  Numerical agreement is NOT a proof."""

import math
import random
import unittest

import mpmath
import numpy as np
import torch

from tests import _path  # noqa: F401
from fxclock import knots as K
from fxclock.models import SPECS, Net
from fxclock.phi import phi12, phi12_naive
from fxclock.prep import collate, global_logsig2, logsig2_windows, prep_window
from fxclock.scan import diag_scan, diag_scan_seq
from fxclock.ssm import DiagSSM
from fxclock.stem import BilinearStem, ExactStem
from n1ref.grids import base_grid, base_times, sample_path, split_grid
from n1ref.model import ToyParams, run_scan
from n1ref.reference import ct_held

torch.set_default_dtype(torch.float64)
CFG = {"n_stem": 8, "stem_tau_min": 0.02, "stem_tau_max": 2.0, "stem_f_max": 25.0, "H": 16, "N": 8, "n_layers": 2,
       "dropout": 0.0, "tf_layers": 2, "tf_heads": 4, "nrde_H": 8, "nrde_hidden": 16}
PATH_FUNCTIONALS = ["P1_foh_clock", "P4_foh_clock", "B_point_clock1", "B_point_clock4", "B_binmean_clock4", "B_rformer_clock4",
                    "B_patch_clock4", "B_tf_clock1", "B_tfpatch_clock4", "B_nrde_clock4"]
PER_OBS = ["A1_foh_perobs", "A3_zoh_perobs", "B_dtonly", "B_bilin_clock4"]  # bilin: not per-obs, but not exact either


def signal(n=128, C=3, dt=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    x = np.stack([np.sin(2 * np.pi * f * t + p) for f, p in [(1.3, 0.2), (7.1, 1.0), (17.0, 2.0)][:C]], 1)
    return K.from_uniform(x + 0.1 * rng.standard_normal((n, C)), dt)


def stem_on(ks, stem, q, rule, t_start=0.0, dtype=torch.float64):
    seg = K.segments(ks, q, rule, t_start)
    T = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt)[None]
    return stem(T(seg["d"]), T(seg["xs"]), T(seg["xe"]), T(seg["win"], torch.long), T(seg["rem"]), T(seg["qdt"]))[0]


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


class TestPhi(unittest.TestCase):
    def ref(self, z):
        mpmath.mp.dps = 50
        zm = mpmath.mpc(z.real, z.imag)
        if zm == 0:
            return complex(1), complex(0.5)
        e = mpmath.exp(zm)
        return complex((e - 1) / zm), complex((e - 1 - zm) / zm ** 2)

    def test_phi_matches_mpmath(self):
        zs = []
        for r in np.logspace(-14, 2.5, 60):
            for ang in np.linspace(math.pi / 2, math.pi, 7):
                zs.append(r * complex(math.cos(ang), math.sin(ang)))
        z = torch.tensor(zs, dtype=torch.complex128)
        p1, p2 = phi12(z)
        for i, zz in enumerate(zs):
            r1, r2 = self.ref(zz)
            self.assertLess(abs(complex(p1[i]) - r1) / abs(r1), 2e-14, msg=f"phi1 at {zz}")
            self.assertLess(abs(complex(p2[i]) - r2) / abs(r2), 2e-14, msg=f"phi2 at {zz}")

    def test_naive_phi2_cancels(self):
        z = torch.tensor([-1e-6 + 0j, -1e-7 + 1e-7j], dtype=torch.complex128)
        _, p2n = phi12_naive(z)
        _, p2 = phi12(z)
        self.assertGreater(float((p2n - p2).abs().max() / 0.5), 1e-4)

    def test_float32_small_argument(self):
        z = torch.tensor([-1e-3, -1e-5, -1e-7], dtype=torch.complex64)
        p1, p2 = phi12(z)
        p1n, p2n = phi12_naive(z)
        self.assertLess(float((p2 - 0.5).abs().max()), 1e-3)
        self.assertGreater(float((p2n - 0.5).abs().max()), 1e-2)

    def test_gradcheck_across_branches(self):
        for r in (1e-8, 0.3, 0.4999, 0.5001, 0.7, 20.0):
            z = torch.tensor([r * complex(-0.8, 0.6)], dtype=torch.complex128, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda v: torch.stack(phi12(v)), (z,), eps=1e-7, atol=1e-6))


class TestScan(unittest.TestCase):
    def test_scan_equals_sequential(self):
        g = torch.Generator().manual_seed(0)
        for L in (1, 2, 7, 64, 129):
            a = torch.exp(torch.complex(-torch.rand(3, L, 5, generator=g), torch.randn(3, L, 5, generator=g)))
            b = torch.complex(torch.randn(3, L, 5, generator=g), torch.randn(3, L, 5, generator=g))
            self.assertLess(rel(diag_scan(a, b), diag_scan_seq(a, b)), 1e-13)

    def test_lti_fft_equals_scan(self):
        layer = DiagSSM(6, 4, dt_ref=0.02, seed=1).double()
        u = torch.randn(2, 100, 6)
        y_lti = layer(u, 0.02)
        y_scan = layer(u, torch.full((2, 100), 0.02))
        self.assertLess(rel(y_lti, y_scan), 1e-10)


class TestStemReferences(unittest.TestCase):
    """Reuses the N1 checks: exact ZOH (n1ref.run_scan zoh_dt) and the independent RK4 reference ct_held."""

    def setUp(self):
        rng = random.Random(1)
        lam = tuple(complex(-math.exp(rng.uniform(math.log(0.2), math.log(2.0))), rng.uniform(0, 4)) for _ in range(4))
        b = tuple(complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(4))
        beta = math.log(math.e - 1.0)                         # softplus(beta) = 1  ->  gate g == 1
        self.p = ToyParams(lam, 0.0, beta, b, (0j,) * 4, (1 + 0j,) * 4, (0j,) * 4)
        self.path = sample_path(rng)
        self.grid = base_grid(base_times(8.0, 16, "jittered", rng=random.Random(9)), self.path)
        self.stem = ExactStem(1, 4, 0.1, 1.0, 1.0, seed=0).double()   # cast BEFORE copying float64 values
        with torch.no_grad():
            self.stem.log_decay.copy_(torch.tensor([math.log(-l.real) for l in lam]))
            self.stem.omega.copy_(torch.tensor([l.imag for l in lam]))
            self.stem.B_re.copy_(torch.tensor([[x.real] for x in b]))
            self.stem.B_im.copy_(torch.tensor([[x.imag] for x in b]))
        # right-held series on the N1 grid: knot k carries u_k = value held on (t_{k-1}, t_k]
        t = np.array(self.grid.times)
        x = np.concatenate([[self.grid.values[0]], self.grid.values])[:, None]
        self.ks = K.KnotSeries(t, x, np.ones(t.shape[0], dtype=bool))

    def test_zoh_stem_equals_n1_exact_zoh(self):
        ref = run_scan(self.p, self.grid.times, self.grid.values, "zoh_dt", 1.0)["h"]
        h = stem_on(self.ks, self.stem, self.ks.t[1:], "zoh")
        ref_t = torch.tensor([list(r) for r in ref[1:]], dtype=torch.complex128)
        self.assertLess(rel(h, ref_t), 1e-13)

    def test_zoh_stem_equals_n1_rk4(self):
        ref = ct_held(self.p, self.grid)["h"]
        h = stem_on(self.ks, self.stem, self.ks.t[1:], "zoh")
        ref_t = torch.tensor([list(r) for r in ref[1:]], dtype=torch.complex128)
        self.assertLess(rel(h, ref_t), 1e-9)

    def test_n1_split_grid_equals_zoh_virtual_refinement(self):
        for m in (2, 4, 8):
            G = split_grid(self.grid, m)
            ks_m = K.refine_uniform(self.ks, m, "zoh")
            self.assertTrue(np.allclose(ks_m.t, G.times, rtol=0, atol=1e-12))
            self.assertTrue(np.array_equal(ks_m.x[1:, 0], np.array(G.values)))
            h1 = stem_on(self.ks, self.stem, self.ks.t[1:], "zoh")
            hm = stem_on(ks_m, self.stem, self.ks.t[1:], "zoh")
            self.assertLess(rel(hm, h1), 1e-13)

    def test_foh_stem_equals_independent_rk4(self):
        lam = self.stem.lam().detach().numpy()
        B = (self.stem.B_re + 1j * self.stem.B_im).detach().numpy()[:, 0]
        q = self.ks.t[1:]
        h = np.zeros(4, dtype=complex)
        out = []
        for k in range(1, self.ks.n):
            t0, t1 = self.ks.t[k - 1], self.ks.t[k]
            x0, x1 = self.ks.x[k - 1, 0], self.ks.x[k, 0]
            n = 400
            dd = (t1 - t0) / n
            X = lambda s: x0 + (s - t0) * (x1 - x0) / (t1 - t0)
            f = lambda s, hh: lam * hh + B * X(s)
            s = t0
            for _ in range(n):
                k1 = f(s, h)
                k2 = f(s + dd / 2, h + dd / 2 * k1)
                k3 = f(s + dd / 2, h + dd / 2 * k2)
                k4 = f(s + dd, h + dd * k3)
                h = h + dd / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                s += dd
            out.append(h.copy())
        ref = torch.tensor(np.array(out))
        got = stem_on(self.ks, self.stem, q, "foh")
        self.assertLess(rel(got, ref), 1e-10)


class TestRefinementInvariance(unittest.TestCase):
    def setUp(self):
        self.ks = signal()
        self.stem = ExactStem(3, 16, 0.02, 2.0, 25.0, seed=3).double()
        self.q1 = K.clock_times(127, 1, 0.02)
        self.q4 = K.clock_times(31, 4, 0.02)

    def test_foh_virtual_knots_exact(self):
        rng = np.random.default_rng(0)
        for q in (self.q1, self.q4):
            h0 = stem_on(self.ks, self.stem, q, "foh")
            for ks2 in (K.refine_uniform(self.ks, 2, "foh"), K.refine_uniform(self.ks, 8, "foh"),
                        K.refine_random(self.ks, 500, "foh", rng)):
                self.assertLess(rel(stem_on(ks2, self.stem, q, "foh"), h0), 1e-13)
            # sanity: a ZOH-valued virtual knot changes the FOH path, so the test can fail
            self.assertGreater(rel(stem_on(K.refine_uniform(self.ks, 2, "zoh"), self.stem, q, "foh"), h0), 1e-4)

    def test_zoh_rule_invariant_under_zoh_virtual_knots(self):
        h0 = stem_on(self.ks, self.stem, self.q4, "zoh")
        self.assertLess(rel(stem_on(K.refine_uniform(self.ks, 4, "zoh"), self.stem, self.q4, "zoh"), h0), 1e-13)
        self.assertGreater(rel(stem_on(K.refine_uniform(self.ks, 4, "foh"), self.stem, self.q4, "zoh"), h0), 1e-4)

    def test_small_dt_extreme_refinement_float64(self):
        # m = 1000 -> segments of 2e-5 s, |lambda d| down to ~1e-6: series branch of phi2 is exercised
        h0 = stem_on(self.ks, self.stem, self.q4, "foh")
        ks2 = K.refine_uniform(self.ks, 1000, "foh")
        self.assertLess(rel(stem_on(ks2, self.stem, self.q4, "foh"), h0), 1e-12)

    def test_models_path_functional_vs_perobs(self):
        ctx = {"t_start": 0.0, "t_end": self.ks.t[-1], "native_dt": 0.02}
        rng = np.random.default_rng(1)
        refined = [K.refine_uniform(self.ks, 4, "foh"), K.refine_random(self.ks, 200, "foh", rng)]
        for name in PATH_FUNCTIONALS + PER_OBS:
            spec = SPECS[name]
            net = Net(spec, 3, 5, CFG, seed=0, native_dt=0.02, ls_scale=np.ones(10) if spec["family"] == "nrde" else None).double().eval()
            with torch.no_grad():
                y0 = net(collate([prep_window(self.ks, spec, ctx)], torch.float64), force_scan=True)
                for ks2 in refined:
                    y = net(collate([prep_window(ks2, spec, ctx)], torch.float64), force_scan=True)
                    if name in PATH_FUNCTIONALS:
                        self.assertLess(rel(y, y0), 1e-12, msg=name)
                    else:
                        self.assertGreater(rel(y, y0), 1e-4, msg=name)

    def test_padding_is_inert(self):
        ctx = {"t_start": 0.0, "t_end": self.ks.t[-1], "native_dt": 0.02}
        short = K.drop_random(self.ks, 0.5, np.random.default_rng(2))
        for name in ("P1_foh_clock", "A1_foh_perobs", "A3_zoh_perobs", "B_dtonly", "B_nrde_clock4"):
            spec = SPECS[name]
            net = Net(spec, 3, 5, CFG, seed=0, native_dt=0.02, ls_scale=np.ones(10) if spec["family"] == "nrde" else None).double().eval()
            with torch.no_grad():
                alone = net(collate([prep_window(short, spec, ctx)], torch.float64), force_scan=True)
                both = net(collate([prep_window(short, spec, ctx), prep_window(self.ks, spec, ctx)], torch.float64),
                           force_scan=True)
            self.assertLess(rel(both[:1], alone), 1e-12, msg=name)


class TestLogsig(unittest.TestCase):
    def test_levy_area_bruteforce_and_invariance(self):
        ks = signal(n=40, C=2, seed=4)
        q = K.clock_times(9, 4, 0.02)
        seg = K.segments(ks, q, "foh", 0.0)
        ls = logsig2_windows(seg, K.evaluate(ks, np.array([0.0]), "foh")[0])
        # brute force: fine Riemann-Stieltjes sums of the area integral on each window
        for j in range(q.shape[0]):
            a = 0.0 if j == 0 else q[j - 1]
            s = np.linspace(a, q[j], 20001)
            Y = np.concatenate([s[:, None], K.evaluate(ks, s, "foh")], axis=1)
            Y0 = Y[0]
            dY = np.diff(Y, axis=0)
            mid = 0.5 * (Y[1:] + Y[:-1]) - Y0
            area = 0.5 * (mid[:, :, None] * dY[:, None, :] - dY[:, :, None] * mid[:, None, :]).sum(0)
            iu = np.triu_indices(3, 1)
            self.assertTrue(np.allclose(ls[j, :3], Y[-1] - Y0, atol=1e-12))
            self.assertTrue(np.allclose(ls[j, 3:], area[iu], atol=1e-9))
        ks2 = K.refine_random(ks, 300, "foh", np.random.default_rng(5))
        seg2 = K.segments(ks2, q, "foh", 0.0)
        ls2 = logsig2_windows(seg2, K.evaluate(ks2, np.array([0.0]), "foh")[0])
        self.assertLess(np.abs(ls2 - ls).max(), 1e-13)


    def test_global_logsig_chen_equals_direct(self):
        ks = signal(n=64, C=2, seed=7)
        q = K.clock_times(15, 4, 0.02)
        x0 = K.evaluate(ks, np.array([0.0]), "foh")[0]
        loc = logsig2_windows(K.segments(ks, q, "foh", 0.0), x0)
        glob = global_logsig2(loc, 3)
        for j in (0, 4, 14):
            direct = logsig2_windows(K.segments(ks, q[j:j + 1], "foh", 0.0), x0)[0]
            self.assertLess(np.abs(glob[j] - direct).max(), 1e-12)


class TestBilinear(unittest.TestCase):
    def test_bilinear_stem_equals_sequential_tustin(self):
        ks = signal(n=40, C=2, seed=8)
        stem = BilinearStem(2, 4, 0.05, 1.0, 5.0, seed=2).double()
        q = K.clock_times(9, 4, 0.02)
        h = stem_on(ks, stem, q, "foh")
        lam = stem.lam().detach().numpy()
        B = (stem.B_re + 1j * stem.B_im).detach().numpy()
        x = np.zeros(4, dtype=complex)
        out = []
        for k in range(1, 37):
            dd = ks.t[k] - ks.t[k - 1]
            z = lam * dd
            x = (1 + z / 2) / (1 - z / 2) * x + dd / (1 - z / 2) * (B @ ks.x[k])
            if k % 4 == 0:
                out.append(x.copy())
        self.assertLess(rel(h, torch.tensor(np.array(out))), 1e-13)


class TestBackward(unittest.TestCase):
    def test_stem_gradcheck(self):
        ks = signal(n=12, C=2, seed=6)
        stem = ExactStem(2, 3, 0.05, 1.0, 5.0, seed=1).double()
        seg = K.segments(ks, K.clock_times(2, 4, 0.02), "foh", 0.0)
        T = lambda a: torch.as_tensor(a)[None]
        d, win, rem, qdt = T(seg["d"]), T(seg["win"]).long(), T(seg["rem"]), T(seg["qdt"])

        def f(xs, xe, ld, om, bre, bim):
            h = torch.func.functional_call(stem, {"log_decay": ld, "omega": om, "B_re": bre, "B_im": bim},
                                           (d, xs, xe, win, rem, qdt))
            return torch.cat([h.real, h.imag], -1)

        args = [T(seg["xs"]), T(seg["xe"])] + [p.detach().clone() for p in
                                               (stem.log_decay, stem.omega, stem.B_re, stem.B_im)]
        args = tuple(x.requires_grad_(True) for x in args)
        self.assertTrue(torch.autograd.gradcheck(f, args, eps=1e-7, atol=1e-6))

    def test_parameter_gradients_refinement_invariant(self):
        ks = signal()
        ctx = {"t_start": 0.0, "t_end": ks.t[-1], "native_dt": 0.02}
        for name in ("P1_foh_clock", "P4_foh_clock", "A1_foh_perobs"):
            spec = SPECS[name]
            net = Net(spec, 3, 5, CFG, seed=0, native_dt=0.02).double().eval()
            grads = []
            for kk in (ks, K.refine_uniform(ks, 4, "foh")):
                net.zero_grad()
                net(collate([prep_window(kk, spec, ctx)], torch.float64), force_scan=True).sum().backward()
                grads.append(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]))
            r = float((grads[1] - grads[0]).abs().max() / grads[0].abs().max())
            if name.startswith("P"):
                self.assertLess(r, 1e-11, msg=name)
            else:
                self.assertGreater(r, 1e-4, msg=name)


class TestKnots(unittest.TestCase):
    def test_flags_and_grid_changes(self):
        ks = signal(n=50)
        r = K.refine_uniform(ks, 3, "foh")
        self.assertEqual(int(r.measured.sum()), 50)
        self.assertEqual(r.n, 50 + 2 * 49)
        s = np.random.default_rng(0).uniform(0, ks.t[-1], 1000)
        self.assertLess(np.abs(K.evaluate(r, s, "foh") - K.evaluate(ks, s, "foh")).max(), 1e-12)
        d = K.downsample(r, 2)
        self.assertTrue(d.measured.all())
        self.assertTrue(np.array_equal(d.t, ks.t[::2]))
        p = K.drop_random(r, 0.5, np.random.default_rng(1))
        self.assertTrue(p.measured.all())
        self.assertEqual(p.t[0], ks.t[0])
        self.assertEqual(p.t[-1], ks.t[-1])

    def test_clock_bit_identical_to_native_times(self):
        ks = signal()
        self.assertTrue(np.array_equal(K.clock_times(127, 1, 0.02), ks.t[1:]))
        self.assertTrue(np.array_equal(K.clock_times(31, 4, 0.02), ks.t[4::4]))


if __name__ == "__main__":
    unittest.main()
