from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from visionguard.data import by_split, index_category  # noqa: E402
from visionguard.features import BACKBONES, FeatureExtractor  # noqa: E402
from visionguard.metrics import au_pro, image_auroc  # noqa: E402
from visionguard.patchcore import (  # noqa: E402
    PatchCore,
    embed_patches,
    gaussian_blur,
    greedy_coreset,
    nearest_neighbours,
    reweighted_image_scores,
)

# ------------------------------------------------------------------ components


def test_embed_patches_shape_and_upsampling():
    feats = [torch.randn(2, 8, 7, 7), torch.randn(2, 16, 4, 4)]
    emb, grid = embed_patches(feats, pretrain_dim=12, target_dim=10)
    assert grid == (7, 7)
    assert emb.shape == (2, 49, 10)
    assert torch.isfinite(emb).all()


def test_embed_patches_constant_input_zero_padded_at_borders():
    """Interior patches keep the value; borders are lower because the 3x3 averaging
    pads with zeros, as in the original PatchCore code and anomalib."""
    feats = [torch.ones(1, 4, 5, 5) * 2.0, torch.ones(1, 6, 3, 3) * 2.0]
    emb, _ = embed_patches(feats, pretrain_dim=8, target_dim=8)
    grid = emb.reshape(5, 5, 8)
    assert torch.allclose(grid[2, 2], torch.full((8,), 2.0))
    assert (grid[0, 0] < 2.0).all()


def _brute_force_k_centre(x: torch.Tensor, n: int, start: int) -> list[int]:
    selected = [start]
    for _ in range(n - 1):
        d = torch.cdist(x, x[selected]).min(dim=1).values
        selected.append(int(torch.argmax(d)))
    return selected


def test_greedy_coreset_matches_brute_force():
    x = torch.randn(200, 6, generator=torch.Generator().manual_seed(1))
    idx = greedy_coreset(x, 15, proj_dim=None, seed=3)
    start = int(idx[0])
    assert idx.tolist() == _brute_force_k_centre(x, 15, start)
    assert len(set(idx.tolist())) == 15


def test_greedy_coreset_covers_better_than_random():
    gen = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(500, 4, generator=gen), torch.randn(5, 4, generator=gen) + 20])
    idx = greedy_coreset(x, 20, proj_dim=None, seed=0)
    rand = torch.randperm(x.shape[0], generator=gen)[:20]

    def cover(sel):
        return torch.cdist(x, x[sel]).min(dim=1).values.max()

    assert cover(idx) < cover(rand)
    assert (idx >= 500).any()  # the far-away cluster is represented


def test_greedy_coreset_with_projection_and_edge_cases():
    x = torch.randn(300, 256)
    idx = greedy_coreset(x, 30, proj_dim=32, seed=0)
    assert len(set(idx.tolist())) == 30
    assert torch.equal(greedy_coreset(x, 30, proj_dim=32, seed=0), idx)  # deterministic
    assert torch.equal(greedy_coreset(x, 400), torch.arange(300))
    with pytest.raises(ValueError):
        greedy_coreset(x, 0)


def test_nearest_neighbours_chunked_matches_full():
    gen = torch.Generator().manual_seed(2)
    q, bank = torch.randn(97, 5, generator=gen), torch.randn(41, 5, generator=gen)
    d, i = nearest_neighbours(q, bank, k=3, max_elements=100)  # forces many chunks
    full = torch.cdist(q, bank)
    ref_d, ref_i = torch.topk(full, 3, dim=1, largest=False)
    assert torch.allclose(d, ref_d, atol=1e-5)
    assert torch.equal(i, ref_i)


def test_reweighting_never_increases_the_max_score():
    gen = torch.Generator().manual_seed(3)
    bank = torch.randn(50, 4, generator=gen)
    emb = torch.randn(3, 10, 4, generator=gen)
    d, i = nearest_neighbours(emb.reshape(30, 4), bank)
    ps, ni = d[:, 0].reshape(3, 10), i[:, 0].reshape(3, 10)
    plain = reweighted_image_scores(ps, ni, emb, bank, num_neighbors=1)
    assert torch.allclose(plain, ps.amax(dim=1))
    weighted = reweighted_image_scores(ps, ni, emb, bank, num_neighbors=9)
    assert (weighted <= plain + 1e-6).all() and (weighted > 0).all()


def test_gaussian_blur_keeps_constants_and_spreads_peaks():
    flat = torch.full((1, 32, 32), 3.0)
    assert torch.allclose(gaussian_blur(flat, 4.0), flat, atol=1e-5)
    spike = torch.zeros(1, 65, 65)
    spike[0, 32, 32] = 1.0
    out = gaussian_blur(spike, 2.0)
    assert out[0, 32, 32] < 1.0 and out[0, 32, 32] == out.max()
    assert out.sum() == pytest.approx(1.0, abs=1e-4)
    assert torch.equal(gaussian_blur(spike, 0.0), spike)


# ------------------------------------------------------------------ backbones


def test_toy_extractor_outputs_two_maps():
    ext = FeatureExtractor("toy")
    out = ext(torch.rand(2, 3, 56, 56))
    assert [tuple(o.shape) for o in out] == [(2, 48, 7, 7), (2, 64, 4, 4)]
    assert not any(p.requires_grad for p in ext.parameters())
    ext.train()
    assert not ext.training


@pytest.mark.parametrize("key", ["wrn50", "vit_sup", "dinov2", "clip"])
def test_real_backbones_give_a_28x28_grid(key):
    pytest.importorskip("timm")
    ext = FeatureExtractor(key, pretrained=False)  # architecture only, no download
    size = ext.preprocess.output_size
    feats = ext(torch.rand(1, 3, size, size))
    emb, grid = embed_patches(feats)
    assert grid == (28, 28)
    assert emb.shape == (1, 784, 1024)
    assert BACKBONES[key].crop / BACKBONES[key].resize == pytest.approx(0.875)


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def structured_mvtec(tmp_path_factory):
    """Normal images: smooth stripes + mild noise. Defects: a dark blob with a mask."""
    root = tmp_path_factory.mktemp("structured")
    rng = np.random.default_rng(0)
    size = 64
    yy, xx = np.mgrid[:size, :size]

    def normal() -> np.ndarray:
        base = 128 + 60 * np.sin(xx / 4.0)
        img = base[..., None] + rng.normal(0, 4, (size, size, 3))
        return np.clip(img, 0, 255).astype(np.uint8)

    def save(arr, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)

    cat = root / "tile"
    for i in range(12):
        save(normal(), cat / "train" / "good" / f"{i:03d}.png")
    for i in range(6):
        save(normal(), cat / "test" / "good" / f"{i:03d}.png")
    for i in range(6):
        img = normal()
        r, c = rng.integers(14, 38, size=2)
        blob = (yy - r - 6) ** 2 + (xx - c - 6) ** 2 < 36
        img[blob] = [20, 200, 20]
        save(img, cat / "test" / "crack" / f"{i:03d}.png")
        save((blob * 255).astype(np.uint8), cat / "ground_truth" / "crack" / f"{i:03d}_mask.png")
    return root


def test_patchcore_finds_planted_defects(structured_mvtec):
    samples = index_category(structured_mvtec, "tile")
    train, test = by_split(samples, "train"), by_split(samples, "test")
    model = PatchCore(FeatureExtractor("toy"), coreset_ratio=0.25, batch_size=4)
    model.fit(train)
    assert model.memory_bank.shape == (round(0.25 * 12 * 49), 1024)
    pred = model.predict(test)
    assert pred.maps.shape == (12, 56, 56) and pred.masks.shape == (12, 56, 56)
    assert pred.ids[0].startswith("tile/test/")
    assert image_auroc(pred.labels, pred.scores) == 1.0
    assert image_auroc(pred.labels, pred.scores_max) == 1.0
    assert au_pro(pred.masks, pred.maps) > 0.5
    # On every defective image, the map peaks on or right next to the defect.
    for amap, mask, label in zip(pred.maps, pred.masks, pred.labels, strict=True):
        if label:
            r, c = np.unravel_index(np.argmax(amap), amap.shape)
            near = mask[max(0, r - 4) : r + 5, max(0, c - 4) : c + 5]
            assert near.any()
    assert pred.seconds_per_image > 0


def test_patchcore_full_bank_and_guards(structured_mvtec):
    samples = index_category(structured_mvtec, "tile")
    train, test = by_split(samples, "train"), by_split(samples, "test")
    model = PatchCore(FeatureExtractor("toy"), batch_size=4)
    with pytest.raises(RuntimeError):
        model.predict(test)
    with pytest.raises(ValueError, match="train/good"):
        model.fit(test)
    model.fit(train[:2], coreset_ratio=None)  # few-shot style: keep every patch
    assert model.memory_bank.shape[0] == 2 * 49
    assert model.memory_bank_mb == pytest.approx(2 * 49 * 1024 * 4 / 2**20)
