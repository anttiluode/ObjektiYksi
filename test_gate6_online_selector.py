from gate6_learned_addressing import addresses
from gate6_online_selector import default_address_index, run_mode


def test_online_default_is_center_155():
    addr = addresses()
    i = default_address_index(addr)
    assert addr[i].source_dr == 0
    assert abs(addr[i].omega - 1.55) < 1e-12


def test_online_selector_only_observes_without_writing():
    run = run_mode(
        "selector_only",
        seed=0,
        n=7,
        epochs=3,
        probes_per_task=3,
        slow_proposals=2,
    )
    r = run["result"]
    assert r["observed_address_fraction"] == 1.0
    assert r["accepted"] == 0
    assert r["changed_edges"] == 0


def test_online_cadaptive_conserves_material():
    run = run_mode(
        "coadaptive_online",
        seed=1,
        n=7,
        epochs=3,
        probes_per_task=3,
        slow_proposals=4,
    )
    r = run["result"]
    assert r["observed_address_fraction"] == 1.0
    assert r["material_sum_relative_error"] < 1e-12
    assert 0 <= r["accepted"] <= r["proposed"]
