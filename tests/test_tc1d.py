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
    calculate_num_steps,
    check_run_type,
    get_history_indices,
    get_particle_temp_and_pressure,
    init_params,
    prep_model,
    run_model,
    set_run_flags,
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
        "debug": "verbose",
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
            Path(__file__).parent.parent / "data" / "tc1d_reference_complete.yaml"
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
            self.PARAM_ALIASES.get(name, name) for name in api_params
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
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                referenced_keys.add(node.value)

        # ero_option1..10 are generated dynamically in _apply_yaml_to_args()
        # using f"ero_option{i}", so they do not appear as literal strings
        # in the function source inspected above.
        referenced_keys.update({f"ero_option{i}" for i in range(1, 11)})

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
            f"CLI arguments missing from the YAML schema: {sorted(missing_from_yaml)}"
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


class TestRunModeFlags:
    """Check the run flags set by prep_model() for each run type."""

    @staticmethod
    def _run_cli(monkeypatch, tmp_path, run_type, erosion_model, general=None):
        """Runs tc1d-cli with a minimal YAML file and returns the params passed
        to run_model() or the batch, NA, or MCMC driver (none of which are run).
        Additional values for the YAML general section can be given in `general`.
        """
        import yaml
        import tc1d.tc1d as tc1d_module

        config = {
            "general": {
                "run_type": run_type,
                **(general or {}),
            },
            "erosion_model": erosion_model,
        }
        input_file = tmp_path / "input.yaml"
        input_file.write_text(yaml.safe_dump(config))

        captured = {}

        def fake_driver(params, batch_params=None):
            captured["params"] = params

        for driver in ["run_model", "batch_run", "batch_run_na", "batch_run_mcmc"]:
            monkeypatch.setattr(tc1d_module, driver, fake_driver)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["tc1d-cli", "--input-file", str(input_file)])
        tc1d_cli.main()
        return captured["params"]

    @pytest.mark.parametrize("run_type", ["na", "mcmc"])
    def test_inverse_mode_with_stage_ranges_only(self, monkeypatch, tmp_path, run_type):
        """Inverse mode should be set for NA/MCMC runs even when the only
        inverted parameters are ero_type 0 stage ranges."""
        erosion_model = {
            "ero_type": 0,
            "ero_stages": [
                {
                    "type": "constant",
                    "unit": "erosion_rate",
                    "duration_myr": 50.0,
                    "parameter1": [0.1, 1.0],
                }
            ],
        }
        params = self._run_cli(monkeypatch, tmp_path, run_type, erosion_model)

        assert params["run_type"] == run_type
        assert params["inverse_mode"] is True
        assert params["batch_mode"] is True
        assert (tmp_path / "csv").is_dir()

    @pytest.mark.parametrize("run_type", ["na", "mcmc"])
    def test_inverse_mode_with_parameter_ranges(self, monkeypatch, tmp_path, run_type):
        erosion_model = {"ero_type": 1, "ero_option1": [5.0, 20.0]}
        params = self._run_cli(monkeypatch, tmp_path, run_type, erosion_model)

        assert params["run_type"] == run_type
        assert params["inverse_mode"] is True
        assert params["batch_mode"] is True

    def test_batch_run_is_not_inverse_mode(self, monkeypatch, tmp_path):
        erosion_model = {"ero_type": 1, "ero_option1": [5.0, 10.0, 20.0]}
        params = self._run_cli(monkeypatch, tmp_path, "batch", erosion_model)

        assert params["run_type"] == "batch"
        assert params["inverse_mode"] is False
        assert params["batch_mode"] is True


class TestCheckRunType:
    """Check that run types that do not match the parameter values are caught."""

    STAGES_WITH_RANGE = [
        {
            "type": "constant",
            "unit": "erosion_rate",
            "duration_myr": 50.0,
            "parameter1": [0.1, 1.0],
        }
    ]

    @staticmethod
    def _params(run_type, ero_type=1, ero_stages_template=None):
        params = init_params(run_type=run_type, ero_type=ero_type)
        params["ero_stages_template"] = ero_stages_template
        return params

    def test_invalid_run_type(self):
        with pytest.raises(ValueError, match="Invalid run type"):
            check_run_type(self._params("inverse"), {"ero_option1": [5.0]})

    def test_forward(self):
        check_run_type(self._params("forward"), {"ero_option1": [5.0]})
        with pytest.raises(ValueError, match="Run type 'forward'.*ero_option1"):
            check_run_type(self._params("forward"), {"ero_option1": [5.0, 10.0]})

    def test_batch(self):
        check_run_type(self._params("batch"), {"ero_option1": [5.0, 10.0, 20.0]})
        with pytest.raises(ValueError, match="Run type 'batch' requires"):
            check_run_type(self._params("batch"), {"ero_option1": [5.0]})

    @pytest.mark.parametrize("run_type", ["na", "mcmc"])
    def test_inversions(self, run_type):
        # Parameter range
        check_run_type(self._params(run_type), {"ero_option1": [5.0, 10.0]})

        # Erosion stage range only (ero_type 0)
        check_run_type(
            self._params(run_type, 0, self.STAGES_WITH_RANGE), {"ero_option1": [0.0]}
        )

        # Nothing to invert
        with pytest.raises(ValueError, match="at least one parameter range"):
            check_run_type(self._params(run_type), {"ero_option1": [5.0]})

        # Range with more than two values
        with pytest.raises(ValueError, match="ero_option1 has 3 values"):
            check_run_type(self._params(run_type), {"ero_option1": [5.0, 10.0, 20.0]})

    def test_prep_model_function_call(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        # Multiple values require run_type="batch"
        params = init_params(run_type="forward", ero_option1=[5.0, 10.0])
        with pytest.raises(ValueError, match="Run type 'forward'"):
            prep_model(params)

        # Inversions are only supported using tc1d-cli
        with pytest.raises(ValueError, match="only supported using tc1d-cli"):
            prep_model(init_params(run_type="na"))


class TestDeprecatedModeFlags:
    """batch_mode and inverse_mode are ignored, but accepted in YAML input files
    for backwards compatibility."""

    RANGES = {"ero_type": 1, "ero_option1": [5.0, 20.0]}

    @pytest.mark.parametrize(
        "run_type, general",
        [
            ("mcmc", {"batch_mode": False, "inverse_mode": True}),
            ("na", {"batch_mode": False, "inverse_mode": True}),
            ("batch", {"batch_mode": True, "inverse_mode": False}),
            ("batch", {"batch_mode": False, "inverse_mode": False}),
        ],
    )
    def test_matching_values_are_accepted_silently(
        self, monkeypatch, tmp_path, capsys, run_type, general
    ):
        params = TestRunModeFlags._run_cli(
            monkeypatch, tmp_path, run_type, self.RANGES, general
        )
        assert params["run_type"] == run_type
        assert params["inverse_mode"] is (run_type in ["na", "mcmc"])
        assert "will be ignored" not in capsys.readouterr().err

    @pytest.mark.parametrize(
        "run_type, general, flag",
        [
            ("mcmc", {"inverse_mode": False}, "inverse_mode=False"),
            ("batch", {"inverse_mode": True}, "inverse_mode=True"),
            ("mcmc", {"batch_mode": True}, "batch_mode=True"),
        ],
    )
    def test_mismatched_values_are_ignored_with_warning(
        self, monkeypatch, tmp_path, capsys, run_type, general, flag
    ):
        params = TestRunModeFlags._run_cli(
            monkeypatch, tmp_path, run_type, self.RANGES, general
        )
        assert params["run_type"] == run_type
        assert params["inverse_mode"] is (run_type in ["na", "mcmc"])
        err = capsys.readouterr().err
        assert flag in err and "will be ignored" in err

    @pytest.mark.parametrize("option", ["--inverse-mode", "--batch-mode"])
    def test_cli_options_removed(self, monkeypatch, capsys, option):
        monkeypatch.setattr("sys.argv", ["tc1d-cli", "--run-type", "mcmc", option])
        with pytest.raises(SystemExit) as exit_info:
            tc1d_cli.main()
        assert exit_info.value.code == 2
        assert f"unrecognized arguments: {option}" in capsys.readouterr().err


class TestSetRunFlags:
    @pytest.mark.parametrize(
        "run_type, batch_mode, inverse_mode",
        [
            ("forward", False, False),
            ("batch", True, False),
            ("na", True, True),
            ("mcmc", True, True),
            ("MCMC", True, True),
        ],
    )
    def test_flags_from_run_type(self, run_type, batch_mode, inverse_mode):
        params = init_params(run_type=run_type)
        set_run_flags(params)
        assert params["batch_mode"] is batch_mode
        assert params["inverse_mode"] is inverse_mode

    def test_forward_run_from_cli(self, monkeypatch, tmp_path):
        params = TestRunModeFlags._run_cli(
            monkeypatch, tmp_path, "forward", {"ero_type": 1, "ero_option1": 5.0}
        )
        assert params["run_type"] == "forward"
        assert params["batch_mode"] is False
        assert params["inverse_mode"] is False


class TestTimeSteps:
    def test_calculate_num_steps(self):
        dt = yr2sec(5000.0)
        assert calculate_num_steps(myr2sec(10.0), dt) == 2000
        # Floating-point error should not change the number of steps
        assert calculate_num_steps(myr2sec(10.0) * (1.0 + 1.0e-12), dt) == 2000
        assert calculate_num_steps(myr2sec(10.0) * (1.0 - 1.0e-12), dt) == 2000
        # Partial time steps are rounded up
        assert calculate_num_steps(myr2sec(10.0), yr2sec(7000.0)) == 1429

    @staticmethod
    def _run_forward(monkeypatch, tmp_path, **kwargs):
        """Runs a short forward model and returns its tTdp history."""
        monkeypatch.chdir(tmp_path)
        params = init_params(
            time=1.0,
            nx=21,
            ero_type=1,
            ero_option1=2.0,
            plot_results=False,
            verbose=0,
            write_ttdp_history=True,
            **kwargs,
        )
        prep_model(params)
        # With past ages, the history for 0 Ma has the surface age in its filename
        history_file = sorted((tmp_path / "csv").glob("time_temp_depth_pressure*.csv"))
        return np.loadtxt(history_file[0], delimiter=",", skiprows=1)

    def test_time_step_not_dividing_model_time(self, monkeypatch, tmp_path):
        # 1 Myr is not a whole number of 7000 year time steps
        with pytest.warns(UserWarning, match="Time step adjusted"):
            history = self._run_forward(monkeypatch, tmp_path, dt=7000.0)

        # Youngest point (first row) is at the surface at 0 Ma
        assert history[0, 0] == pytest.approx(0.0)
        assert history[0, 2] == pytest.approx(0.0, abs=1.0)

    def test_past_age_increment_not_dividing_time_steps(self, monkeypatch, tmp_path):
        # With past ages every 0.35 Myr in a 1 Myr model, the surface times are 0,
        # 0.35, and 0.7 Ma. The 0.35 Ma particle reaches the surface after 0.65 Myr,
        # which is not a whole number of 4000 year time steps (162.5).
        self._run_forward(monkeypatch, tmp_path, dt=4000.0, past_age_increment=0.35)
        history = np.loadtxt(
            tmp_path / "csv" / "time_temp_depth_pressure_hist_0.35Ma.csv",
            delimiter=",",
            skiprows=1,
        )
        # The history is rounded up to whole time steps (163 steps), and starts
        # at the start of the model and ends at the surface
        assert history[0, 2] == pytest.approx(0.0, abs=1.0)
        assert history[-1, 0] == pytest.approx(163 * 0.004)


class TestThermalHistories:
    def test_get_history_indices(self):
        # Youngest point first, every write_increment-th point, oldest included
        assert get_history_indices(11, 2) == [10, 8, 6, 4, 2, 0]
        assert get_history_indices(10, 2) == [9, 7, 5, 3, 1, 0]
        assert get_history_indices(500, 50)[0] == 499
        assert get_history_indices(500, 50)[-1] == 0
        assert get_history_indices(1, 2) == [0]

    def test_history_covers_full_model_time(self, monkeypatch, tmp_path):
        # 1 Myr with 200 time steps and 2 km of erosion
        history = TestTimeSteps._run_forward(monkeypatch, tmp_path, dt=5000.0)

        # Youngest point (first row) is at the surface at 0 Ma
        assert history[0, 0] == pytest.approx(0.0)
        assert history[0, 2] == pytest.approx(0.0, abs=1.0)

        # Oldest point (last row) is the initial state at the start of the model
        assert history[-1, 0] == pytest.approx(1.0)
        assert history[-1, 2] == pytest.approx(2000.0, rel=1.0e-6)


class TestParticleTempAndPressure:
    @staticmethod
    def _interp(slope):
        return lambda depth: slope * depth

    def test_surface_temperature(self):
        # Particles at the surface have the surface temperature and zero pressure
        temp, pressure = get_particle_temp_and_pressure(
            0.0, 35.0e3, False, 10.0, self._interp(0.02), self._interp(3.0e4)
        )
        assert temp == 10.0
        assert pressure == 0.0

    def test_temperature_at_depth(self):
        temp, pressure = get_particle_temp_and_pressure(
            5.0e3, 35.0e3, False, 10.0, self._interp(0.02), self._interp(3.0e4)
        )
        assert temp == pytest.approx(100.0)
        assert pressure == pytest.approx(1.5e8)

    def test_fixed_moho(self):
        # Particles below a fixed Moho have the Moho temperature and pressure
        temp, pressure = get_particle_temp_and_pressure(
            40.0e3, 35.0e3, True, 10.0, self._interp(0.02), self._interp(3.0e4)
        )
        assert temp == pytest.approx(700.0)
        assert pressure == pytest.approx(1.05e9)


class TestPastAges:
    def test_past_sample_reaches_surface_at_surface_time(self, monkeypatch, tmp_path):
        # 1 Myr model with 2 km of erosion and a past surface time of 0.5 Ma
        TestTimeSteps._run_forward(
            monkeypatch, tmp_path, dt=5000.0, past_age_increment=0.5
        )
        history = np.loadtxt(
            tmp_path / "csv" / "time_temp_depth_pressure_hist_0.5Ma.csv",
            delimiter=",",
            skiprows=1,
        )

        # Youngest point (first row) is at the surface at the surface time
        # (times in the history are relative to the surface time)
        assert history[0, 0] == pytest.approx(0.0)
        assert history[0, 2] == pytest.approx(0.0, abs=1.0)

        # Oldest point is at the start of the model, 0.5 Myr before the
        # surface time, at the depth eroded in that time (1 km)
        assert history[-1, 0] == pytest.approx(0.5)
        assert history[-1, 2] == pytest.approx(1000.0, rel=1.0e-6)

    def test_measured_ages_use_present_day_history(self, monkeypatch, tmp_path):
        """Measured ages without depositional ages should be compared to the
        present-day sample, also when past ages are calculated."""
        monkeypatch.chdir(tmp_path)
        obs_age_file = Path(__file__).parent.parent / "data" / "sample_data.csv"

        def misfit(**kwargs):
            params = init_params(
                time=2.0,
                dt=5000.0,
                nx=21,
                ero_type=1,
                ero_option1=4.0,
                obs_age_file=str(obs_age_file),
                plot_results=False,
                verbose=0,
                **kwargs,
            )
            return run_model(params)

        assert misfit(past_age_increment=0.5) == pytest.approx(misfit())


class TestCrustalUplift:
    """The Moho should move with the crust in crustal uplift models, unless
    the Moho is fixed."""

    @staticmethod
    def _final_moho_depth(monkeypatch, tmp_path, capsys, **kwargs):
        monkeypatch.chdir(tmp_path)
        params = init_params(
            time=1.0,
            dt=5000.0,
            init_moho_depth=50.0,
            ero_type=1,
            ero_option1=2.0,
            calc_ages=False,
            plot_results=False,
            echo_info=False,
            **kwargs,
        )
        prep_model(params)
        for line in capsys.readouterr().out.splitlines():
            if line.startswith("- Final Moho depth:"):
                return float(line.split()[-2])
        raise AssertionError("Final Moho depth not found in model output")

    @pytest.mark.parametrize("nx", [51, 201])
    @pytest.mark.parametrize("mantle_velocity", [0.0, 0.5])
    def test_moho_moves_with_crust(
        self, monkeypatch, tmp_path, capsys, nx, mantle_velocity
    ):
        moho_depth = self._final_moho_depth(
            monkeypatch,
            tmp_path,
            capsys,
            nx=nx,
            crustal_uplift=True,
            mantle_velocity=mantle_velocity,
        )
        # 2 km of erosion moves the Moho from 50 to 48 km
        assert moho_depth == pytest.approx(48.0)

    def test_fixed_moho(self, monkeypatch, tmp_path, capsys):
        moho_depth = self._final_moho_depth(
            monkeypatch, tmp_path, capsys, nx=51, crustal_uplift=True, fixed_moho=True
        )
        assert moho_depth == pytest.approx(50.0)


class TestPastAgeSurfaceTimes:
    @staticmethod
    def _surface_times(monkeypatch, tmp_path, time, past_age_increment):
        """Returns the surface times (Ma) written to past_ages.csv."""
        monkeypatch.chdir(tmp_path)
        params = init_params(
            time=time,
            dt=5000.0,
            nx=21,
            ero_type=1,
            ero_option1=2.0,
            past_age_increment=past_age_increment,
            write_past_ages=True,
            plot_results=False,
            verbose=0,
        )
        prep_model(params)
        past_ages = np.loadtxt(tmp_path / "csv" / "past_ages.csv", delimiter=",")
        return np.atleast_2d(past_ages)[:, 0]

    @pytest.mark.parametrize(
        "time, increment, expected",
        [
            (1.0, 0.2, [0.8, 0.6, 0.4, 0.2, 0.0]),
            # Increments that do not divide the model duration
            (1.0, 0.3, [0.9, 0.6, 0.3, 0.0]),
            (1.0, 0.4, [0.8, 0.4, 0.0]),
            # Floating-point division of 0.3 / 0.1 gives 2.9999999999999996
            (0.3, 0.1, [0.2, 0.1, 0.0]),
        ],
    )
    def test_surface_times(self, monkeypatch, tmp_path, time, increment, expected):
        surface_times = self._surface_times(monkeypatch, tmp_path, time, increment)
        assert surface_times == pytest.approx(expected)

    def test_increment_longer_than_model(self, monkeypatch, tmp_path):
        with pytest.warns(UserWarning, match="Only present-day ages"):
            surface_times = self._surface_times(monkeypatch, tmp_path, 1.0, 1.5)
        assert surface_times == pytest.approx([0.0])
