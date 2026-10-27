"""Tests for thermochronometer age calculations (using tcages)."""

import numpy as np
import pytest
import tcages

from tc1d.tc1d import calculate_ages_and_tcs, ft_ages, he_ages, init_params, myr2sec

# Linear cooling from 250 °C to 5 °C over 10 Myr with 2001 model time steps
N_STEPS = 2001
TIME_HISTORY = np.linspace(0.0, myr2sec(10.0), N_STEPS)  # Seconds since model start
TEMP_HISTORY = np.linspace(250.0, 5.0, N_STEPS)
ZEROS = np.zeros(N_STEPS)

# Order of values returned by calculate_ages_and_tcs()
OUTPUT_NAMES = [
    "corr_ahe_age",
    "ahe_age",
    "ahe_temp",
    "aft_age",
    "aft_mean_ftl",
    "aft_temp",
    "corr_zhe_age",
    "zhe_age",
    "zhe_temp",
    "zft_age",
    "zft_temp",
]

# Reference values for the history above with default parameters, calculated
# with Tc1D 0.6.0. These are identical to the values from Tc1D 0.5.0, which
# used its own builds of the RDAAM and ketch codes.
REFERENCE = {
    "corr_ahe_age": 2.5689239501953125,
    "ahe_age": 1.9182968139648438,
    "ahe_temp": 67.93863677978516,
    "aft_age": 4.444220682137267,
    "aft_mean_ftl": 13.960710718804542,
    "aft_temp": 113.88340671236304,
    "corr_zhe_age": 4.8090362548828125,
    "zhe_age": 3.8649368286132812,
    "zhe_temp": 122.82138824462892,
    "zft_age": 9.433681207174121,
    "zft_temp": 236.12518957576597,
}


def run_age_calculation(write_track_lengths=False, **param_updates):
    """Calculates ages for the test history and returns them as a dict."""
    params = init_params(
        echo_inputs=False, echo_info=False, echo_thermal_info=False, echo_ages=False
    )
    params.update(ero_type=1, pad_time=0.0)
    params.update(param_updates)
    results = calculate_ages_and_tcs(
        params,
        TIME_HISTORY,
        TEMP_HISTORY,
        ZEROS,
        ZEROS,
        False,
        "",
        False,
        "",
        write_track_lengths,
    )
    return dict(zip(OUTPUT_NAMES, (float(r) for r in results)))


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path, monkeypatch):
    """Runs each test in an empty temporary directory."""
    monkeypatch.chdir(tmp_path)


def test_ages_match_reference():
    results = run_age_calculation()
    for name, expected in REFERENCE.items():
        assert results[name] == pytest.approx(expected, rel=1e-3), name


def test_age_order():
    results = run_age_calculation()
    assert (
        results["corr_ahe_age"]
        < results["aft_age"]
        < results["corr_zhe_age"]
        < results["zft_age"]
    )


def test_padding_succeeds_and_only_affects_zhe():
    """Padding used to repeat the oldest time point, which made RDAAM fail.

    With 2001 steps the output interval divides the history evenly, which
    triggered the failure. Padding at the starting temperature (250 °C) should
    only change the ZHe age, as the other systems are fully reset.
    """
    unpadded = run_age_calculation()
    padded = run_age_calculation(pad_time=10.0)
    assert padded["corr_ahe_age"] == pytest.approx(unpadded["corr_ahe_age"])
    assert padded["aft_age"] == pytest.approx(unpadded["aft_age"])
    assert padded["corr_zhe_age"] > unpadded["corr_zhe_age"]


def test_track_length_file(tmp_path):
    results = run_age_calculation(write_track_lengths=True)
    lines = (tmp_path / "ft_length.csv").read_text().splitlines()
    assert lines[0] == "Track length (um),Probability"
    data = np.loadtxt(tmp_path / "ft_length.csv", delimiter=",", skiprows=1)
    assert data.shape == (200, 2)
    assert data[0, 0] == pytest.approx(0.05)
    assert data[-1, 0] == pytest.approx(19.95)
    # Probability density integrates to 1 over 0.1 um bins
    assert data[:, 1].sum() * 0.1 == pytest.approx(1.0, abs=1e-3)
    # Mean length from the distribution matches the reported mean length
    mean_length = (data[:, 0] * data[:, 1]).sum() * 0.1
    assert mean_length == pytest.approx(results["aft_mean_ftl"], rel=1e-3)


def test_wrappers_match_tcages():
    times = [0.0, 5.0, 10.0]
    temps = [5.0, 100.0, 250.0]
    grains = dict(
        ap_rad=60.0,
        ap_uranium=10.0,
        ap_thorium=40.0,
        zr_rad=60.0,
        zr_uranium=100.0,
        zr_thorium=40.0,
    )
    he = tcages.he_ages(
        times,
        temps,
        ap_rad=60.0,
        ap_u=10.0,
        ap_th=40.0,
        zr_rad=60.0,
        zr_u=100.0,
        zr_th=40.0,
    )
    assert he_ages(times, temps, **grains) == (he.ahe, he.ahe_corr, he.zhe, he.zhe_corr)
    aft = tcages.aft_age(times, temps)
    assert ft_ages(times, temps, write_track_lengths=False) == (
        aft.age,
        aft.mean_length,
    )


def test_failed_age_calculation_raises():
    """A failed RDAAM calculation raises an exception instead of exiting."""
    with pytest.raises(tcages.TcagesError):
        he_ages(
            [0.0, 5.0, 10.0, 10.0],
            [5.0, 100.0, 250.0, 250.0],
            ap_rad=60.0,
            ap_uranium=10.0,
            ap_thorium=40.0,
            zr_rad=60.0,
            zr_uranium=100.0,
            zr_thorium=40.0,
        )
