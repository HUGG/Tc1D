import numpy as np
from tc1d.tc1d import yr2sec, myr2sec, kilo2base, milli2base, micro2base
from tc1d.tc1d import mmyr2ms, deg2rad, round_to_base, calculate_eu

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
- calculate_erosion_rate
- calculate_exhumation_magnitude

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
