from gate6_moving_target import run_mode, stress_addresses


def test_stress_address_bank_is_3x5():
    addr = stress_addresses()
    assert len(addr) == 15
    assert len({(a.source_dr, a.omega) for a in addr}) == 15


def test_stress_selector_only_keeps_material_frozen():
    run = run_mode(
        "selector_only",
        seed=0,
        n=7,
        epochs=2,
        slow_proposals=3,
        probes_per_task=1,
    )
    r = run["result"]
    assert r["changed_edges"] == 0
    assert r["accepted"] == 0
    assert r["material_sum_relative_error"] == 0.0


def test_stress_cadaptive_conserves_material():
    run = run_mode(
        "coadaptive",
        seed=1,
        n=7,
        epochs=3,
        slow_proposals=4,
        probes_per_task=1,
    )
    r = run["result"]
    assert r["material_sum_relative_error"] < 1e-12
    assert 0 <= r["accepted"] <= r["proposed"]
    assert r["oracle_address_switches"] >= 0
    assert r["inactive_damage_epochs"] >= 0
