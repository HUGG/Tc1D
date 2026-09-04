import numpy as np
import pytest
import argparse
import ast
import inspect
import textwrap
import tc1d.tc1d_cli as tc1d_cli
from pathlib import Path
from tc1d.tc1d import yr2sec, myr2sec, kilo2base, milli2base, micro2base
from tc1d.tc1d import (
    mmyr2ms,
    deg2rad,
    round_to_base,
    calculate_eu,
    read_ero_stages_from_yaml,
    erosion_constant,
    erosion_linear,
    erosion_exponential,
    calculate_erosion_rate,
    calculate_exhumation_magnitude,
    init_params
)
from tc1d.tc1d_cli import (
    _apply_yaml_to_args,
    _validate_yaml_keys,
    _load_yaml_dict,
    YAML_ALLOWED_KEYS,
    YAML_INVERSION_ALLOWED_KEYS,
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
* read_ero_stages_from_yaml (ero_type=0: erosion_rate + thickness + errors)
* format_ero_stages_table (ero_type=0: YAML echo strings)
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

    def test_read_ero_stages_from_yaml_erosion_rate(self):
        raw = [
            {
                "type": "constant",
                "unit": "erosion_rate",
                "duration_myr": 5.0,
                "parameter1": 0.1,
            },
            {
                "type": "linear",
                "unit": "erosion_rate",
                "duration_myr": 5.0,
                "parameter1": 0.1,
                "parameter2": 0.3,
            },
            {
                "type": "exponential",
                "unit": "erosion_rate",
                "duration_myr": 5.0,
                "parameter1": 0.3,
                "parameter2": 2.0,
                "parameter3": 0.1,
            },
        ]

        stages = read_ero_stages_from_yaml(raw)
        assert len(stages) == 3
        assert stages[0]["type"] == "constant"
        assert stages[0]["unit"] == "erosion_rate"
        assert stages[0]["duration_myr"] == 5.0
        assert stages[0]["input_params"] == [0.1, None, None]
        assert stages[2]["input_params"] == [0.3, 2.0, 0.1]

    def test_read_ero_stages_from_yaml_thickness(self):
        raw = [
            {
                "type": "constant",
                "unit": "thickness",
                "duration_myr": 5.0,
                "parameter1": 0.5,
            },
            {
                "type": "linear",
                "unit": "thickness",
                "duration_myr": 5.0,
                "parameter1": 1.0,
                "parameter2": 0.6,
            },
            {
                "type": "exponential",
                "unit": "thickness",
                "duration_myr": 5.0,
                "parameter1": 1.5,
                "parameter2": 2.0,
                "parameter3": 0.5,
            },
        ]

        stages = read_ero_stages_from_yaml(raw)
        assert len(stages) == 3
        assert stages[1]["input_params"] == [1.0, 0.6, None]
        assert stages[2]["input_params"] == [1.5, 2.0, 0.5]

    def test_read_ero_stages_from_yaml_errors(self):
        # tau <= 0 should fail
        raw = [
            {
                "type": "exponential",
                "unit": "erosion_rate",
                "duration_myr": 5.0,
                "parameter1": 0.1,
                "parameter2": 0.0,
                "parameter3": 0.2,
            }
        ]
        with pytest.raises(ValueError):
            read_ero_stages_from_yaml(raw)

        # s out of bounds should fail
        raw2 = [
            {
                "type": "linear",
                "unit": "thickness",
                "duration_myr": 5.0,
                "parameter1": 1.0,
                "parameter2": 1.2,
            }
        ]
        with pytest.raises(ValueError):
            read_ero_stages_from_yaml(raw2)

    def test_read_ero_stages_from_yaml_bad_float(self):
        raw = [
            {
                "type": "constant",
                "unit": "erosion_rate",
                "duration_myr": 5.0,
                "parameter1": "abc",
            }
        ]
        with pytest.raises(ValueError):
            read_ero_stages_from_yaml(raw)

    # Test: calculate_exhumation_magnitude for type0 constant rate (no truncation): r_const * duration.
    def test_calculate_exhumation_magnitude_type0_constant_rate(self):
        raw = [
            {
                "type": "constant",
                "unit": "erosion_rate",
                "duration_myr": 10.0,
                "parameter1": 0.2,
            }
        ]
        stages = read_ero_stages_from_yaml(raw)

        t_total_sec = myr2sec(10.0)
        mag_km, fw = calculate_exhumation_magnitude(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            t_total_sec,
            stages,
        )
        assert abs(mag_km - 2.0) < 1e-10
        assert fw is False

    # Test: calculate_exhumation_magnitude truncates integration when total stage duration exceeds t_total.
    def test_calculate_exhumation_magnitude_type0_truncation_constant(self):
        raw = [
            {
                "type": "constant",
                "unit": "erosion_rate",
                "duration_myr": 10.0,
                "parameter1": 0.2,
            }
        ]
        stages = read_ero_stages_from_yaml(raw)

        t_total_sec = myr2sec(5.0)
        mag_km, fw = calculate_exhumation_magnitude(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            t_total_sec,
            stages,
        )
        assert abs(mag_km - 1.0) < 1e-10
        assert fw is False

    # Test: calculate_exhumation_magnitude for type0 linear rate matches analytic integral (mean rate * duration).
    def test_calculate_exhumation_magnitude_type0_linear_rate(self):
        raw = [
            {
                "type": "linear",
                "unit": "erosion_rate",
                "duration_myr": 10.0,
                "parameter1": 0.0,
                "parameter2": 1.0,
            }
        ]
        stages = read_ero_stages_from_yaml(raw)

        t_total_sec = myr2sec(10.0)
        mag_km, fw = calculate_exhumation_magnitude(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            t_total_sec,
            stages,
        )
        assert abs(mag_km - 5.0) < 1e-10
        assert fw is False

    # Test: calculate_exhumation_magnitude for type0 exponential rate matches analytic integral.
    def test_calculate_exhumation_magnitude_type0_exponential_rate(self):
        raw = [
            {
                "type": "exponential",
                "unit": "erosion_rate",
                "duration_myr": 10.0,
                "parameter1": 1.0,
                "parameter2": 2.0,
                "parameter3": 0.0,
            }
        ]
        stages = read_ero_stages_from_yaml(raw)

        t_total_sec = myr2sec(10.0)
        mag_km, fw = calculate_exhumation_magnitude(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            t_total_sec,
            stages,
        )

        expected = 2.0 * (1.0 - np.exp(-5.0))
        assert abs(mag_km - expected) < 1e-10
        assert fw is False

    # Test: calculate_erosion_rate returns zero after the last stage when sum(stage durations) < t_total.
    def test_calculate_erosion_rate_tail_to_zero(self):
        # One stage: 5 Myr at 0.2 km/Myr, but model time longer -> tail=0
        raw = [
            {
                "type": "constant",
                "unit": "erosion_rate",
                "duration_myr": 5.0,
                "parameter1": 0.2,
            }
        ]
        stages = read_ero_stages_from_yaml(raw)

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
            params,
            dt_sec,
            t_total_sec,
            current_time,
            x,
            vx_array,
            0.0,
            moho_depth,
            False,
            0.0,
        )

        assert abs(vx_surf - 0.0) < 1e-30

    # Test: calculate_erosion_rate switches stages correctly around a stage boundary (no off-by-one).
    def test_calculate_erosion_rate_stage_switching(self):
        raw = [
            {
                "type": "constant",
                "unit": "erosion_rate",
                "duration_myr": 5.0,
                "parameter1": 0.2,
            },
            {
                "type": "constant",
                "unit": "erosion_rate",
                "duration_myr": 5.0,
                "parameter1": 0.0,
            },
        ]
        stages = read_ero_stages_from_yaml(raw)

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
            params,
            dt_sec,
            t_total_sec,
            myr2sec(4.999),
            x,
            vx_array,
            0.0,
            moho_depth,
            False,
            0.0,
        )
        assert abs(vx_surf - mmyr2ms(0.2)) < 1e-25

        # Just after boundary (stage 2)
        vx_array, vx_surf, vx_max, fault_depth = calculate_erosion_rate(
            params,
            dt_sec,
            t_total_sec,
            myr2sec(5.001),
            x,
            vx_array,
            0.0,
            moho_depth,
            False,
            0.0,
        )
        assert abs(vx_surf - 0.0) < 1e-30

class TestConfigurationConsistency:

    PARAM_ALIASES = {
        "length": "max_depth",
        "time": "t_total",
        "mantle_adiabat": "mantle_adiabat",
        "calc_ages": "calc_ages",
        "plot_results": "plot_results",
        "display_plots": "display_plots",
        "plot_ma": "plot_ma",
    }

    def test_yaml_additional_cli_mappings(self):
        """
        Test that YAML parameters with corresponding CLI arguments are
        correctly transferred to the argparse Namespace.
        """

        # Minimal argparse-like namespace containing the parameters
        # tested here.
        args = argparse.Namespace(
            mantle_velocity=[0.0],
            plot_density=False,
            plot_elevation_history=False,
            plot_peclet_number=False,
            plot_ft_length_dist=False,
        )

        # Minimal YAML-like dictionary.
        yaml_config = {
            "erosion_model": {
                "mantle_velocity": 1.5,
            },
            "plotting": {
                "plot_density": True,
                "plot_elevation_history": True,
                "plot_peclet_number": True,
                "plot_ft_length_dist": True,
            },
        }

        _apply_yaml_to_args(args, yaml_config)

        assert args.mantle_velocity == [1.5]
        assert args.plot_density is True
        assert args.plot_elevation_history is True
        assert args.plot_peclet_number is True
        assert args.plot_ft_length_dist is True

    def test_yaml_unknown_key_raises_error(self):
        yaml_config = {
            "thermal_model": {
                "explicit": False,
            }
        }

        with pytest.raises(ValueError, match="explicit"):
            _validate_yaml_keys(yaml_config)

    def test_yaml_unknown_section_raises_error(self):
        yaml_config = {
            "old_thermal_options": {
                "temp_base": 1300.0,
            }
        }

        with pytest.raises(ValueError, match="old_thermal_options"):
            _validate_yaml_keys(yaml_config)

    def test_reference_yaml_schema_is_valid(self):
        """
        Test that the complete reference YAML only contains sections and keys
        supported by the current tc1d-cli YAML interface.
        """

        yaml_path = (
                Path(__file__).parent.parent
                / "data"
                / "tc1d_reference_complete.yaml"
        )

        yaml_config = _load_yaml_dict(yaml_path)

        # Raises ValueError if an unsupported section or key is present.
        _validate_yaml_keys(yaml_config)

    def test_init_params_matches_internal_parameter_dict(self):
        """
        Check that user-facing parameters defined by tc1d.init_params()
        are represented in the internal Tc1D parameter dictionary.
        """

        # Get argument names from the Python API.
        signature = inspect.signature(init_params)
        api_params = set(signature.parameters.keys())

        # Translate intentionally different API/internal names.
        translated_api_params = {
            self.PARAM_ALIASES.get(name, name)
            for name in api_params
        }

        # Get the actual internal parameter keys returned by init_params().
        actual_params = set(init_params().keys())

        # Every API parameter should have an internal representation.
        missing = translated_api_params - actual_params

        assert not missing, (
            "Parameters defined in init_params() are missing from the "
            f"internal params dictionary: {sorted(missing)}"
        )

    def _get_cli_params_dict_keys(self):
        """
        Extract the keys of the params dictionary built inside tc1d_cli.main()
        without executing the CLI.
        """

        source = inspect.getsource(tc1d_cli.main)
        tree = ast.parse(textwrap.dedent(source))

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "params":
                        if isinstance(node.value, ast.Dict):
                            return {
                                key.value
                                for key in node.value.keys
                                if isinstance(key, ast.Constant)
                                   and isinstance(key.value, str)
                            }

        raise AssertionError(
            "Could not find the params dictionary inside tc1d_cli.main()."
        )

    def _get_cli_argument_dests(self):
        """
        Extract argparse destination names defined in tc1d_cli.main()
        without executing the CLI.
        """

        source = inspect.getsource(tc1d_cli.main)
        tree = ast.parse(textwrap.dedent(source))

        dests = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Look only for calls such as:
            # group.add_argument(...)
            # parser.add_argument(...)
            if not (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
            ):
                continue

            # Prefer an explicit argparse dest=...
            dest = None

            for keyword in node.keywords:
                if (
                        keyword.arg == "dest"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                ):
                    dest = keyword.value.value
                    break

            # If dest= is absent, derive it from the long option.
            # Example: --debug -> debug
            if dest is None:
                long_options = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant)
                       and isinstance(arg.value, str)
                       and arg.value.startswith("--")
                ]

                if long_options:
                    dest = long_options[0][2:].replace("-", "_")

            if dest is not None:
                dests.add(dest)

        return dests

    def _get_yaml_keys_referenced_by_apply_function(self):
        """
        Extract YAML key names referenced inside _apply_yaml_to_args()
        without executing the function.
        """

        source = inspect.getsource(_apply_yaml_to_args)
        tree = ast.parse(textwrap.dedent(source))

        referenced_keys = set()

        # Collect string constants appearing in the function.
        # YAML keys are explicitly referenced either in:
        #   if "key" in section:
        # or in tuples such as:
        #   for k in ("key1", "key2", ...)
        for node in ast.walk(tree):
            if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
            ):
                referenced_keys.add(node.value)

        # ero_option1..10 are generated dynamically in _apply_yaml_to_args()
        # using f"ero_option{i}", so they do not appear as literal strings
        # in the function source inspected above.
        referenced_keys.update(
            {f"ero_option{i}" for i in range(1, 11)}
        )

        return referenced_keys

    def test_cli_params_match_internal_params(self):
        """
        Check that the parameter dictionary built by tc1d-cli is consistent
        with the internal parameter dictionary returned by init_params().
        """

        cli_params = self._get_cli_params_dict_keys()
        internal_params = set(init_params().keys())

        # Parameters intentionally created only by the CLI.
        cli_only = {
            "cmd_line_call",
            "batch_mode",
            "inverse_mode",
            "ero_stages_template",
        }

        missing_from_cli = internal_params - cli_params
        unexpected_in_cli = cli_params - internal_params - cli_only

        assert not missing_from_cli, (
            "Internal Tc1D parameters missing from tc1d-cli params dictionary: "
            f"{sorted(missing_from_cli)}"
        )

        assert not unexpected_in_cli, (
            "Unexpected parameters in tc1d-cli params dictionary: "
            f"{sorted(unexpected_in_cli)}"
        )

    def test_cli_arguments_are_represented_in_yaml_schema(self):
        """
        Check that user-configurable CLI arguments are represented in the
        YAML schema, except for arguments intentionally specific to the CLI.
        """

        cli_args = self._get_cli_argument_dests()

        yaml_keys = set()

        for section, keys in YAML_ALLOWED_KEYS.items():
            if section == "inversion":
                continue

            yaml_keys.update(keys)

        for keys in YAML_INVERSION_ALLOWED_KEYS.values():
            yaml_keys.update(keys)

        cli_only = {
            "version",
            "input_file",
        }

        missing_from_yaml = cli_args - yaml_keys - cli_only

        assert not missing_from_yaml, (
            "CLI arguments missing from the YAML schema: "
            f"{sorted(missing_from_yaml)}"
        )

    def test_yaml_schema_keys_are_applied_by_cli(self):
        """
        Check that every key declared valid in the YAML schema is referenced
        by _apply_yaml_to_args(), preventing valid YAML keys from being
        silently ignored.
        """

        referenced_keys = self._get_yaml_keys_referenced_by_apply_function()

        # Flatten all standard YAML keys.
        schema_keys = set()

        for section, keys in YAML_ALLOWED_KEYS.items():
            if section == "inversion":
                continue

            schema_keys.update(keys)

        # Add the nested inversion parameter keys.
        for keys in YAML_INVERSION_ALLOWED_KEYS.values():
            schema_keys.update(keys)

        missing_from_apply = schema_keys - referenced_keys

        assert not missing_from_apply, (
            "YAML schema keys not handled by _apply_yaml_to_args(): "
            f"{sorted(missing_from_apply)}"
        )