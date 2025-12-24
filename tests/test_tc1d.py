import numpy as np
import pytest
from tc1d.tc1d import yr2sec, myr2sec, kilo2base, milli2base, micro2base
from tc1d.tc1d import (
    mmyr2ms, deg2rad, round_to_base, calculate_eu,
    read_ero_file, format_ero_stages_table,
    erosion_constant, erosion_linear, erosion_exponential,
    calculate_erosion_rate, calculate_exhumation_magnitude
)

"""
List of tests to still create:
# Conversions
* yr2sec
* myr2sec
* kilo2base
* milli2base
* micro2base
* mmyr2ms
* deg2rad
* round_to_base
- tt_hist_to_ma

# Thermal
- calculate_heat_flow
- calculate_explicit_stability
- adiabat
- temp_ss_implicit
- temp_transient_explicit
- temp_transient_implicit
- create_intrusion
- apply_intrusion

# Density
- calculate_pressure
- update_density
- calculate_isostatic_elevation

# Materials
- update_materials?
- calculate_crust_solidus
- calculate_mantle_solidus

# Chronometers
- calculate_eu
- he_ages
- ft_ages
- calculate_closure_temp
- calculate_ages_and_tcs
- calculate_misfit

# Erosion
- init_ero_types?
* erosion_constant
* erosion_linear
* erosion_exponential
* read_ero_file (ero_type=0: erosion_rate + thickness + errors)
* format_ero_stages_table (ero_type=0: CSV echo strings)
* calculate_exhumation_magnitude (ero_type=0: constant + truncation)
* calculate_exhumation_magnitude (ero_type=0: linear)
* calculate_exhumation_magnitude (ero_type=0: exponential)
* calculate_erosion_rate (ero_type=0: tail-to-zero)
* calculate_erosion_rate (ero_type=0: stage switching boundaries)

# Plotting
- plot_predictions_no_data?
- plot_predictions_with_data?
- plot_measurements?

# InputOutput
- get_write_increment
- write_tt_history?
- write_ttdp_history?
- read_age_data_file
- create_output_directory?
- log_output?

# Inversion
- def objective
- log_prior
- log_likelihood
- log_probability

# Structure
- check_execs?
- init_params?
- prep_model?
- batch_run?
- batch_run_na?
- batch_run_mcmc?
- run_model?
"""


class TestConversions:
    def test_yr2sec(self):
        year_in_seconds = 31557600.0
        assert yr2sec(time_yr=1.0) == year_in_seconds

    def test_myr2sec(self):
        myr_in_seconds = 31557600000000.0
        assert myr2sec(time_myr=1.0) == myr_in_seconds

    def test_kilo2base(self):
        kilo = 1000.0
        assert kilo2base(value=1.0) == kilo

    def test_milli2base(self):
        milli = 1.0e-3
        assert milli2base(value=1.0) == milli

    def test_micro2base(self):
        micro = 1.0e-6
        assert micro2base(value=1.0) == micro

    def test_mmyr2ms(self):
        mmyr_in_ms = 3.168808781e-11
        assert round(mmyr2ms(rate=1.0), 20) == mmyr_in_ms

    def test_deg2rad(self):
        deg2rad_test_value1 = np.pi
        deg2rad_test_value2 = np.pi / 3.0
        deg2rad_test_value3 = 2.0 * np.pi
        assert round(deg2rad(value=180.0), 20) == round(deg2rad_test_value1, 20)
        assert round(deg2rad(value=60.0), 20) == round(deg2rad_test_value2, 20)
        assert round(deg2rad(value=360.0), 20) == round(deg2rad_test_value3, 20)

    def test_round_to_base(self):
        round_to_base_test_value1 = 750.0
        round_to_base_test_value2 = 10.0
        round_to_base_test_value3 = 5000.0
        assert round_to_base(x=747.39, base=50) == round_to_base_test_value1
        assert round_to_base(x=14.9, base=10) == round_to_base_test_value2
        assert round_to_base(x=4500.1, base=1000) == round_to_base_test_value3


# class TestThermal:


class TestChronometers:
    def test_calculate_eu(self):
        calculate_eu_test_value1 = 100.0
        calculate_eu_test_value2 = 23.8
        calculate_eu_test_value3 = 247.6
        assert (
            round(calculate_eu(uranium=100.0, thorium=0.0), 10)
            == calculate_eu_test_value1
        )
        assert (
            round(calculate_eu(uranium=0.0, thorium=100.0), 10)
            == calculate_eu_test_value2
        )
        assert (
            round(calculate_eu(uranium=200.0, thorium=200.0), 10)
            == calculate_eu_test_value3
        )


class TestErosionType0:
    # Test: constant stage law returns the constant rate regardless of time.
    def test_erosion_constant(self):
        assert erosion_constant(0.0, 0.12) == 0.12
        assert erosion_constant(5.0, -0.03) == -0.03

    # Test: linear stage law interpolates correctly at start, mid, and end of the stage.
    def test_erosion_linear(self):
        assert erosion_linear(0.0, 0.0, 1.0, 10.0) == 0.0
        assert erosion_linear(10.0, 0.0, 1.0, 10.0) == 1.0
        assert erosion_linear(5.0, 0.0, 1.0, 10.0) == 0.5

    # Test: exponential stage law equals r_start at t=0 and approaches r_target for large time.
    def test_erosion_exponential(self):
        r = erosion_exponential(0.0, 1.0, 0.0, 2.0)
        assert abs(r - 1.0) < 1e-12

        r = erosion_exponential(1e6, 1.0, 0.0, 2.0)
        assert abs(r - 0.0) < 1e-12

    # Test: read_ero_file parses erosion_rate-mode stages and preserves raw CSV params (csv_params).
    def test_read_ero_file_erosion_rate(self, tmp_path):
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "constant,erosion_rate,5.0,0.1,,\n"
            "linear,erosion_rate,5.0,0.1,0.3,\n"
            "exponential,erosion_rate,5.0,0.3,2.0,0.1\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)

        stages = read_ero_file(str(f))
        assert len(stages) == 3
        assert stages[0]["type"] == "constant"
        assert stages[0]["unit"] == "erosion_rate"
        assert stages[0]["duration_myr"] == 5.0
        assert stages[0]["csv_params"] == [0.1, None, None]
        assert stages[2]["csv_params"] == [0.3, 2.0, 0.1]

    # Test: read_ero_file parses thickness-mode stages and preserves raw CSV params (csv_params).
    def test_read_ero_file_thickness(self, tmp_path):
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "constant,thickness,5.0,0.5,,\n"
            "linear,thickness,5.0,1.0,0.6,\n"
            "exponential,thickness,5.0,1.5,2.0,0.5\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)

        stages = read_ero_file(str(f))
        assert len(stages) == 3
        assert stages[1]["csv_params"] == [1.0, 0.6, None]
        assert stages[2]["csv_params"] == [1.5, 2.0, 0.5]

    # Test: read_ero_file raises ValueError for invalid tau (<=0) and out-of-bounds shape factor s.
    def test_read_ero_file_errors(self, tmp_path):
        # tau <= 0 should fail
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "exponential,erosion_rate,5.0,0.1,0.0,0.2\n"
        )
        f = tmp_path / "ero_bad.csv"
        f.write_text(csv_text)
        with pytest.raises(ValueError):
            read_ero_file(str(f))

        # s out of bounds should fail
        csv_text2 = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "linear,thickness,5.0,1.0,1.2,\n"
        )
        f2 = tmp_path / "ero_bad2.csv"
        f2.write_text(csv_text2)
        with pytest.raises(ValueError):
            read_ero_file(str(f2))

    # Test: read_ero_file raises ValueError for non-float numeric fields (robust parsing).
    def test_read_ero_file_bad_float(self, tmp_path):
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "constant,erosion_rate,5.0,abc,,\n"
        )
        f = tmp_path / "ero_bad.csv"
        f.write_text(csv_text)
        with pytest.raises(ValueError):
            read_ero_file(str(f))

    # Test: format_ero_stages_table includes expected headers/fields and the "unitless" convention string.
    def test_format_ero_stages_table_contains_expected_fields(self, tmp_path):
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "linear,thickness,5.0,1.0,0.6,\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)

        stages = read_ero_file(str(f))
        table = format_ero_stages_table(stages)

        assert "Stage" in table
        assert "linear" in table
        assert "thickness" in table
        assert "p2=s (unitless)" in table

    # Test: calculate_exhumation_magnitude for type0 constant rate (no truncation): r_const * duration.
    def test_calculate_exhumation_magnitude_type0_constant_rate(self, tmp_path):
        # r_const=0.2 km/Myr over 10 Myr => 2.0 km
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "constant,erosion_rate,10.0,0.2,,\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)

        t_total_sec = myr2sec(10.0)
        mag_km, fw = calculate_exhumation_magnitude(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            t_total_sec,
            str(f),
        )
        assert abs(mag_km - 2.0) < 1e-10
        assert fw is False

    # Test: calculate_exhumation_magnitude truncates integration when total stage duration exceeds t_total.
    def test_calculate_exhumation_magnitude_type0_truncation_constant(self, tmp_path):
        # r_const=0.2 km/Myr over 10 Myr, but t_total=5 Myr => 1.0 km
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "constant,erosion_rate,10.0,0.2,,\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)

        t_total_sec = myr2sec(5.0)
        mag_km, fw = calculate_exhumation_magnitude(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            t_total_sec,
            str(f),
        )
        assert abs(mag_km - 1.0) < 1e-10
        assert fw is False

    # Test: calculate_exhumation_magnitude for type0 linear rate matches analytic integral (mean rate * duration).
    def test_calculate_exhumation_magnitude_type0_linear_rate(self, tmp_path):
        # linear rate 0 -> 1 km/Myr over 10 Myr => mean 0.5 => 5.0 km
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "linear,erosion_rate,10.0,0.0,1.0,\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)

        t_total_sec = myr2sec(10.0)
        mag_km, fw = calculate_exhumation_magnitude(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            t_total_sec,
            str(f),
        )
        assert abs(mag_km - 5.0) < 1e-10
        assert fw is False

    # Test: calculate_exhumation_magnitude for type0 exponential rate matches analytic integral.
    def test_calculate_exhumation_magnitude_type0_exponential_rate(self, tmp_path):
        # r(t) = r_target + (r_start-r_target)*exp(-t/tau)
        # Here: r_start=1, r_target=0, tau=2 Myr, dt=10 Myr
        # Expected thickness = 2*(1-exp(-10/2)) = 2*(1-exp(-5))
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "exponential,erosion_rate,10.0,1.0,2.0,0.0\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)

        t_total_sec = myr2sec(10.0)
        mag_km, fw = calculate_exhumation_magnitude(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            t_total_sec,
            str(f),
        )

        expected = 2.0 * (1.0 - np.exp(-5.0))
        assert abs(mag_km - expected) < 1e-10
        assert fw is False

    # Test: calculate_erosion_rate returns zero after the last stage when sum(stage durations) < t_total.
    def test_calculate_erosion_rate_tail_to_zero(self, tmp_path):
        # One stage: 5 Myr at 0.2 km/Myr, but model time longer -> tail=0
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "constant,erosion_rate,5.0,0.2,,\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)
        stages = read_ero_file(str(f))

        params = {
            "ero_type": 0,
            "ero_stages": stages,
            "ero_total_stage_sec": sum(st["dt_sec"] for st in stages),
            "crustal_uplift": False,
        }

        x = np.array([0.0, 1000.0])
        vx_array = np.zeros_like(x)

        t_total_sec = myr2sec(20.0)
        dt_sec = myr2sec(0.01)
        current_time = myr2sec(10.0)
        moho_depth = 0.0

        vx_array, vx_surf, vx_max, fault_depth = calculate_erosion_rate(
            params, dt_sec, t_total_sec, current_time,
            x, vx_array, 0.0, moho_depth, False
        )

        assert abs(vx_surf - 0.0) < 1e-30

    # Test: calculate_erosion_rate switches stages correctly around a stage boundary (no off-by-one).
    def test_calculate_erosion_rate_stage_switching(self, tmp_path):
        csv_text = (
            "type,unit,duration_myr,parameter1,parameter2,parameter3\n"
            "constant,erosion_rate,5.0,0.2,,\n"
            "constant,erosion_rate,5.0,0.0,,\n"
        )
        f = tmp_path / "ero.csv"
        f.write_text(csv_text)
        stages = read_ero_file(str(f))

        params = {
            "ero_type": 0,
            "ero_stages": stages,
            "ero_total_stage_sec": sum(st["dt_sec"] for st in stages),
            "crustal_uplift": False,
        }

        x = np.array([0.0, 1000.0])
        vx_array = np.zeros_like(x)
        t_total_sec = myr2sec(20.0)
        dt_sec = myr2sec(0.001)  # small dt to reduce boundary ambiguity
        moho_depth = 0.0

        # Just before boundary (still stage 1)
        vx_array, vx_surf, vx_max, fault_depth = calculate_erosion_rate(
            params, dt_sec, t_total_sec, myr2sec(4.999),
            x, vx_array, 0.0, moho_depth, False
        )
        assert abs(vx_surf - mmyr2ms(0.2)) < 1e-25

        # Just after boundary (stage 2)
        vx_array, vx_surf, vx_max, fault_depth = calculate_erosion_rate(
            params, dt_sec, t_total_sec, myr2sec(5.001),
            x, vx_array, 0.0, moho_depth, False
        )
        assert abs(vx_surf - 0.0) < 1e-30