from ising.runs import config_path, temperature_grid, temperature_label


def test_temperature_grid_has_clean_values():
    temperatures = temperature_grid(0.5, 4.0, 0.1)
    assert len(temperatures) == 36
    assert temperature_label(temperatures[27]) == "3.2000"  # 1/(1/3.2) would not round-trip
    assert len({temperature_label(T) for T in temperatures}) == len(temperatures)


def test_config_path_depends_on_formatted_temperature_only():
    assert config_path("wolff", 16, 3.2) == config_path("wolff", 16, 3.200000000000002)
    assert config_path("wolff", 16, 3.2) != config_path("wolff", 32, 3.2)