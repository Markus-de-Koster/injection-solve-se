"""Round-trip + persistence tests for the encodings and the HarmonicNormalizer."""

from __future__ import annotations

import pytest
import torch

from injection_hse.encoding import CartesianEncoding, TrigEncoding
from injection_hse.normalization import HarmonicNormalizer

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _fake_state(seed: int = 0) -> torch.Tensor:
    """Heavy-tailed per-harmonic complex state ``[B, H, N]`` (fundamental >> high orders)."""
    torch.manual_seed(seed)
    v = (torch.randn(64, 7, 12) + 1j * torch.randn(64, 7, 12)).to(torch.complex64)
    scale = torch.tensor([1.0, 0.1, 0.05, 0.02, 0.01, 0.005, 0.003]).view(1, 7, 1)
    return v * scale


@pytest.mark.parametrize("Enc", [TrigEncoding, CartesianEncoding])
def test_normalize_inverse_is_identity(Enc):
    enc = Enc()
    v = _fake_state()
    feats = enc.encode(v).transpose(1, 2)  # [B, N, H, C]
    norm = HarmonicNormalizer().fit(feats, magnitude_transform="log1p")

    normalized = norm.transform(feats)
    recovered = norm.inverse_transform(normalized)
    assert torch.allclose(recovered, feats, atol=1e-4), (
        (recovered - feats).abs().max().item()
    )


@pytest.mark.parametrize("Enc", [TrigEncoding, CartesianEncoding])
def test_encode_decode_then_normalize_recovers_complex(Enc):
    enc = Enc()
    v = _fake_state()
    feats = enc.encode(v).transpose(1, 2)
    norm = HarmonicNormalizer().fit(feats)
    # full pipeline: encode -> normalize -> inverse -> decode -> complex
    rec_v = enc.decode(norm.inverse_transform(norm.transform(feats))).transpose(1, 2)
    assert torch.allclose(rec_v, v, atol=1e-4), (rec_v - v).abs().max().item()


def test_state_dict_round_trip_is_exact():
    enc = TrigEncoding()
    feats = enc.encode(_fake_state()).transpose(1, 2)
    norm = HarmonicNormalizer().fit(feats)
    target = norm.transform(feats)

    restored = HarmonicNormalizer()
    restored.load_state_dict(norm.state_dict())
    assert torch.equal(restored.transform(feats), target)
    assert torch.equal(restored.cartesian_scale, norm.cartesian_scale)


def test_loss_standardize_equalizes_harmonics():
    """After per-harmonic standardize, each order's Cartesian residual has ~unit scale."""
    enc = CartesianEncoding()
    v = _fake_state()
    feats = enc.encode(v).transpose(1, 2)  # [B, N, H, 2]
    norm = HarmonicNormalizer().fit(feats)
    cart = enc.to_cartesian(feats)  # [B, N, H, 2]
    std = norm.loss_standardize(cart)  # divided by per-harmonic scale
    # RMS over (re, im) and the batch should be O(1) for every order (not 1e-3 spread)
    rms = std.pow(2).mean(dim=(0, 1, 3)).sqrt()  # [H]
    assert (rms > 0.1).all() and (rms < 10.0).all(), rms.tolist()


def test_cos_sin_channels_pass_through_for_trig():
    enc = TrigEncoding()
    feats = enc.encode(_fake_state()).transpose(1, 2)
    norm = HarmonicNormalizer().fit(feats)
    # cos/sin channels (1, 2) are passed through: mean 0 / std 1, no transform
    assert torch.equal(norm.mean[:, 1:], torch.zeros_like(norm.mean[:, 1:]))
    assert torch.equal(norm.std[:, 1:], torch.ones_like(norm.std[:, 1:]))
    # magnitude channel (0) IS standardized (non-trivial mean)
    assert norm.mean[:, 0].abs().sum() > 0


@pytest.mark.parametrize("Enc", [TrigEncoding, CartesianEncoding])
def test_per_harmonic_true_is_byte_identical_default(Enc):
    """``per_harmonic=True`` is the default and reproduces the prior fit byte-for-byte."""
    enc = Enc()
    feats = enc.encode(_fake_state()).transpose(1, 2)
    default = HarmonicNormalizer().fit(feats)
    explicit = HarmonicNormalizer().fit(feats, per_harmonic=True)
    assert default.per_harmonic is True
    assert torch.equal(default.mean, explicit.mean)
    assert torch.equal(default.std, explicit.std)
    assert torch.equal(default.cartesian_scale, explicit.cartesian_scale)


@pytest.mark.parametrize("Enc", [TrigEncoding, CartesianEncoding])
def test_per_harmonic_false_shares_one_scale_across_orders(Enc):
    """``per_harmonic=False`` fits ONE shared statistic, repeated across the harmonic axis."""
    enc = Enc()
    feats = enc.encode(_fake_state()).transpose(1, 2)  # [B, N, H, C]
    shared = HarmonicNormalizer().fit(feats, per_harmonic=False)
    assert shared.per_harmonic is False

    # every order gets the SAME standardization stats and the SAME Cartesian loss scale
    assert torch.allclose(shared.mean, shared.mean[:1].expand_as(shared.mean))
    assert torch.allclose(shared.std, shared.std[:1].expand_as(shared.std))
    assert torch.allclose(
        shared.cartesian_scale,
        shared.cartesian_scale[:1].expand_as(shared.cartesian_scale),
    )

    # and it genuinely differs from the per-harmonic fit (the heavy-tailed spectrum spans
    # orders of magnitude, so a single shared scale is not the per-order scale)
    per_h = HarmonicNormalizer().fit(feats, per_harmonic=True)
    assert not torch.allclose(shared.cartesian_scale, per_h.cartesian_scale)

    # the transform stays invertible under the shared statistics (shape unchanged)
    assert shared.transform(feats).shape == feats.shape
    assert torch.allclose(
        shared.inverse_transform(shared.transform(feats)), feats, atol=1e-4
    )


def test_inverse_transform_survives_drifted_passthrough_channels():
    """The expm1 in inverse_transform must not NaN the backward when a DISCARDED
    (pass-through cos/sin) channel is large: torch.where evaluates both branches,
    and an overflowed expm1 in the discarded lane makes its backward 0*inf = NaN."""
    import torch

    from injection_hse.normalization import HarmonicNormalizer

    norm = HarmonicNormalizer().fit(torch.rand(32, 5, 3, 3))
    feats = torch.zeros(2, 5, 3, 3)
    feats[..., 1] = 200.0  # drifted cos channel (decode-scale-invariant)
    feats[..., 2] = -150.0  # drifted sin channel
    feats.requires_grad_(True)
    out = norm.inverse_transform(feats)
    assert bool(torch.isfinite(out).all())
    out.sum().backward()
    assert bool(torch.isfinite(feats.grad).all())
