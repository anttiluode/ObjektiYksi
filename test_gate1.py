import numpy as np

from gate1_route_or_resonance import mirror_couplings, spectral_metrics
from self_carving import make_grid, train


def test_mirroring_learned_material_swaps_target_preference():
    run = train("phase_reference", epochs=35, eta=0.035)
    grid = run["grid"]
    g = run["g"]
    m = mirror_couplings(grid, g)
    direct = spectral_metrics(grid, g, 1.55)["desired_over_decoy"]
    mirrored = spectral_metrics(grid, m, 1.55)["desired_over_decoy"]
    assert direct > 1.0
    assert mirrored < 1.0
    assert abs(direct * mirrored - 1.0) < 1e-7


def test_material_mirror_preserves_budget_distribution():
    run = train("phase_reference", epochs=8, eta=0.02)
    grid = run["grid"]
    g = run["g"]
    m = mirror_couplings(grid, g)
    assert np.allclose(np.sort(g), np.sort(m))
    assert abs(np.mean(m) - 1.0) < 1e-12
