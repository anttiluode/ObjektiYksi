from gate7_address_ablation import address_spaces, run_space


def test_gate7_spaces_remove_one_address_dimension():
    spaces = address_spaces()
    assert len(spaces["full"]) == 9
    assert len(spaces["position_only"]) == 3
    assert len(spaces["frequency_only"]) == 3
    assert {a.omega for a in spaces["position_only"]} == {1.35}
    assert {a.source_dr for a in spaces["frequency_only"]} == {0}


def test_gate7_small_runs_conserve_material():
    for name in ("full", "position_only", "frequency_only"):
        run = run_space(
            name,
            seed=0,
            n=7,
            epochs=3,
            probes_per_task=3,
            slow_proposals=4,
        )
        r = run["result"]
        assert r["observed_address_fraction"] == 1.0
        assert r["material_sum_relative_error"] < 1e-12
        assert 0 <= r["accepted"] <= r["proposed"]
