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
    calculate_ages_and_tcs,
    calculate_misfit,
    calculate_num_steps,
    check_run_type,
    disable_output_for_inversions,
    get_history_indices,
    get_pad_times,
    get_tt_path,
    get_particle_temp_and_pressure,
    init_params,
    prep_model,
    run_model,
    save_ttdp_history,
    set_aft_model,
    set_run_flags,
)
from tc1d.tc1d_cli import (
    _apply_yaml_to_args,
    _cli_warning,
    _validate_yaml_keys,
    _load_yaml_dict,
    YAML_ALLOWED_KEYS,
    YAML_INVERSION_ALLOWED_KEYS,
)

"""
List of functions with tests (*) and tests still to create (-). Tests for the
age calculations are in test_ages.py.

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
* get_particle_temp_and_pressure

# Density
- calculate_pressure
- update_density
- calculate_isostatic_elevation

# Materials
- update_materials?
- calculate_crust_solidus
- calculate_mantle_solidus

# Chronometers
* calculate_eu
* he_ages (test_ages.py)
* ft_ages (test_ages.py)
- calculate_closure_temp
* calculate_ages_and_tcs (also test_ages.py)
* set_aft_model
* calculate_misfit (misfit type 2 only)

# Erosion
- init_ero_types?
* erosion_constant
* erosion_linear
* erosion_exponential
* read_ero_stages_from_yaml (ero_type=0: erosion_rate + thickness + errors)
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
* get_history_indices
* get_pad_times
* get_tt_path
- save_tt_history
* save_ttdp_history
- read_age_data_file (only tested indirectly)
- create_output_directory?
- log_output? (log file contents tested in TestLogFile)

# Inversion
- def objective
- log_prior
- log_likelihood
- log_probability

# Structure
- check_execs?
- init_params?
* check_run_type
* set_run_flags
* disable_output_for_inversions
- prep_model? (partly tested through other tests)
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
    def _run_cli(
        monkeypatch, tmp_path, run_type, erosion_model, general=None, output=None
    ):
        """Runs tc1d-cli with a minimal YAML file and returns the params passed
        to run_model() or the batch, NA, or MCMC driver (none of which are run).
        Additional values for the YAML general and output sections can be given
        in `general` and `output`.
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
        if output is not None:
            config["output"] = output
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


class TestCrashFixes:
    """Input combinations that previously crashed."""

    @staticmethod
    def _run(monkeypatch, tmp_path, **kwargs):
        monkeypatch.chdir(tmp_path)
        model = dict(
            time=1.0,
            dt=20000.0,
            nx=21,
            ero_type=1,
            ero_option1=2.0,
            plot_results=False,
            verbose=0,
        )
        model.update(kwargs)
        params = init_params(**model)
        prep_model(params)
        return params

    def test_ero_type2_integer_time(self, monkeypatch, tmp_path):
        self._run(
            monkeypatch,
            tmp_path,
            time=1,
            ero_type=2,
            ero_option1=1.0,
            ero_option2=0.5,
            ero_option3=1.0,
        )

    def test_single_value_lists(self, monkeypatch, tmp_path):
        params = self._run(monkeypatch, tmp_path, ero_option1=[2.0])
        assert params["ero_option1"] == 2.0

    def test_tuples_for_batch_runs(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, run_type="batch", ero_option1=(1.0, 2.0))
        log = (tmp_path / "csv" / "tc1d_batch_log.csv").read_text().splitlines()
        assert len(log) == 3  # header and two models

    @pytest.mark.parametrize("run_type", ["forward", "batch"])
    def test_log_output_without_ages(self, monkeypatch, tmp_path, run_type):
        ero_option2 = [0.0, 1.0] if run_type == "batch" else 0.0
        params = self._run(
            monkeypatch,
            tmp_path,
            run_type=run_type,
            ero_option2=ero_option2,
            calc_ages=False,
            log_output=True,
        )
        log = (tmp_path / "csv" / params["log_file"]).read_text().splitlines()
        # Header plus one row per model, with -9999.0 for the age values
        assert len(log) == (3 if run_type == "batch" else 2)
        assert "-9999.0000" in log[1]

    @pytest.mark.parametrize("depo_age", [1.0, 2.0])
    def test_depositional_age_not_younger_than_model(
        self, monkeypatch, tmp_path, depo_age
    ):
        # Depositional ages equal to or older than the model start (1 Ma)
        age_file = tmp_path / "ages.csv"
        age_file.write_text(
            "Age type, Age (Ma), Standard deviation (Ma), eU concentration (ppm), "
            "Grain radius (um), Sample ID, Depositional age (Ma)\n"
            f"AHe, 9.0, 0.5, 40.0, 60.0, S1, {depo_age}\n"
        )
        with pytest.raises(
            ValueError, match="must be younger than the model starting time"
        ):
            self._run(monkeypatch, tmp_path, obs_age_file=str(age_file))

    def test_age_file_with_blank_lines(self, monkeypatch, tmp_path):
        # Blank lines between samples and at the end of the file
        age_file = tmp_path / "ages.csv"
        age_file.write_text(
            "Age type, Age (Ma), Standard deviation (Ma), eU concentration (ppm), "
            "Grain radius (um), Sample ID, Depositional age (Ma)\n"
            "AHe, 9.0, 0.5, 40.0, 60.0, S1,\n"
            "\n"
            "ZHe, 28.0, 1.5, 900.0, 60.0, S1,\n"
            "\n"
            "   \n"
        )
        self._run(
            monkeypatch, tmp_path, obs_age_file=str(age_file), write_age_output=True
        )
        summary = (tmp_path / "csv" / "age_summary.csv").read_text().splitlines()
        assert len(summary) == 3  # header and the two samples

    def test_na_continues_after_failed_model(self, monkeypatch, tmp_path):
        """Models that fail during an NA inversion should be rejected, not stop
        the inversion. The search and appraisal are replaced by simple stand-ins
        so that only the objective function is tested."""
        import matplotlib
        import yaml
        import tc1d.tc1d as tc1d_module

        matplotlib.use("Agg")

        # Forward model stand-in: fails for ero_option1 > 5, otherwise returns
        # ero_option1 as the misfit
        def fake_run_model(params):
            if params["ero_option1"] > 5.0:
                raise ValueError("test model failure")
            return params["ero_option1"]

        class FakeSearcher:
            def __init__(self, objective, ns, nr, ni, n, bounds):
                self.objective = objective
                self.ni = ni
                self.bounds = bounds

            def run(self):
                self.samples = np.array(
                    [[2.0, 1.0], [8.0, 2.0], [4.0, 3.0], [9.0, 1.5], [3.0, 2.5]]
                )
                self.objectives = np.array([self.objective(x) for x in self.samples])

        class FakeAppraiser:
            def __init__(self, initial_ensemble, log_ppd, bounds, **kwargs):
                self.samples = initial_ensemble
                self.mean = initial_ensemble.mean(axis=0)
                self.sample_mean_error = np.zeros(2)
                self.covariance = np.cov(initial_ensemble.T)
                self.sample_covariance_error = np.zeros((2, 2))

            def run(self):
                pass

        monkeypatch.setattr(tc1d_module, "run_model", fake_run_model)
        monkeypatch.setattr(tc1d_module, "NASearcher", FakeSearcher)
        monkeypatch.setattr(tc1d_module, "NAAppraiser", FakeAppraiser)
        monkeypatch.chdir(tmp_path)

        config = {
            "general": {"run_type": "na", "verbose": 0},
            "erosion_model": {
                "ero_type": 1,
                "ero_option1": [1.0, 10.0],
                "ero_option2": [0.0, 5.0],
            },
            "plotting": {"no_display_plots": True},
        }
        input_file = tmp_path / "input.yaml"
        input_file.write_text(yaml.safe_dump(config))
        monkeypatch.setattr("sys.argv", ["tc1d-cli", "--input-file", str(input_file)])
        tc1d_cli.main()

        # Failed models (ero_option1 = 8 and 9) get a large misfit
        results = np.loadtxt(
            tmp_path / "csv" / "na_searcher_results.csv", delimiter=",", skiprows=1
        )
        np.testing.assert_allclose(results[:, -1], [2.0, 1.0e10, 4.0, 1.0e10, 3.0])


class TestMisfit:
    def test_misfit_type2(self):
        # Sum of squared normalized residuals divided by (N - num_params - 1)
        measured = np.array([10.0, 12.0, 14.0])
        predicted = np.array([11.0, 12.0, 12.0])
        stdev = np.array([1.0, 1.0, 1.0])
        assert calculate_misfit(predicted, measured, stdev, 2, 1) == pytest.approx(5.0)

    @pytest.mark.parametrize("n_ages, num_params", [(1, 0), (2, 1), (2, 2)])
    def test_misfit_type2_too_few_ages(self, n_ages, num_params):
        ages = np.linspace(10.0, 20.0, n_ages)
        with pytest.raises(ValueError, match="Misfit type 2 requires more"):
            calculate_misfit(ages + 1.0, ages, ages / 10.0, 2, num_params)


class TestLogFile:
    """Every row in the log file should have one value per header column."""

    OBS_AGE_FILE = str(Path(__file__).parent.parent / "data" / "sample_data.csv")

    @staticmethod
    def _read_log(log_file):
        import csv

        rows = list(csv.reader(log_file.open()))
        header, data = rows[0], rows[1:]
        assert len(data) > 0
        for row in data:
            assert len(row) == len(header)
        return header, data

    def test_forward_log(self, monkeypatch, tmp_path):
        params = TestCrashFixes._run(
            monkeypatch,
            tmp_path,
            temp_surf=5.0,
            log_output=True,
            model_id="RUN1",
        )
        header, data = self._read_log(tmp_path / "csv" / params["log_file"])
        values = dict(zip(header, data[0]))
        assert values["Model ID"] == "RUN1"
        assert values["Node points"] == "21"
        assert float(values["Surface temperature (C)"]) == 5.0
        assert float(values["Total exhumation (km)"]) == pytest.approx(2.0)

    def test_batch_log_with_failed_model(self, monkeypatch, tmp_path):
        # The explicit solution (type 1) is unstable for these grid and time steps,
        # but the batch run should continue and log both models
        TestCrashFixes._run(
            monkeypatch,
            tmp_path,
            run_type="batch",
            nx=101,
            dt=50000.0,
            solution_type=[1, 2],
        )
        _, data = self._read_log(tmp_path / "csv" / "tc1d_batch_log.csv")
        assert [row[0] for row in data] == ["M0001", "M0002"]

    @pytest.mark.parametrize("run_type", ["na", "mcmc"])
    def test_inversion_log(self, monkeypatch, tmp_path, run_type):
        import matplotlib
        import yaml

        matplotlib.use("Agg")
        config = {
            "general": {"run_type": run_type, "verbose": 0},
            "geometry_time": {"time": 1.0, "dt": 20000.0, "nx": 21},
            "erosion_model": {
                "ero_type": 2,
                "ero_option1": [0.5, 2.0],
                "ero_option2": 0.5,
                "ero_option3": [0.5, 2.0],
            },
            "observations": {"obs_age_file": self.OBS_AGE_FILE},
            "plotting": {"no_plot_results": True, "no_display_plots": True},
            "inversion": {
                "neighbourhood_algorithm": {
                    "na_ns": 2,
                    "na_nr": 1,
                    "na_ni": 2,
                    "na_n": 1,
                    "na_n_resample": 4,
                    "na_n_walkers": 1,
                },
                "mcmc": {
                    "mcmc_nwalkers": 4,
                    "mcmc_nsteps": 1,
                    "mcmc_discard": 0,
                    "mcmc_thin": 1,
                },
            },
        }
        input_file = tmp_path / "input.yaml"
        input_file.write_text(yaml.safe_dump(config))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["tc1d-cli", "--input-file", str(input_file)])
        tc1d_cli.main()

        _, data = self._read_log(tmp_path / "csv" / "tc1d_batch_log.csv")
        # Models run during inversions are not given model IDs
        assert set(row[0] for row in data) == {""}

    @pytest.mark.parametrize("age_types", [["AHe", "ZHe"], ["ZHe"]])
    def test_measured_ages_in_log(self, monkeypatch, tmp_path, age_types):
        """Measured ages from an age data file are written to the matching log
        columns, including when there are no AHe ages."""
        ages = {
            "AHe": "AHe, 9.0, 0.5, 40.0, 60.0, S1,\n",
            "ZHe": "ZHe, 28.0, 1.5, 900.0, 60.0, S1,\n",
        }
        age_file = tmp_path / "ages.csv"
        age_file.write_text(
            "Age type, Age (Ma), Standard deviation (Ma), eU concentration (ppm), "
            "Grain radius (um), Sample ID, Depositional age (Ma)\n"
            + "".join(ages[age_type] for age_type in age_types)
        )
        params = TestCrashFixes._run(
            monkeypatch, tmp_path, obs_age_file=str(age_file), log_output=True
        )
        header, data = self._read_log(tmp_path / "csv" / params["log_file"])
        values = dict(zip(header, data[0]))
        if "AHe" in age_types:
            assert float(values["Measured apatite (U-Th)/He age (Ma)"]) == 9.0
        else:
            assert float(values["Measured apatite (U-Th)/He age (Ma)"]) == -9999.0
        assert float(values["Measured zircon (U-Th)/He age (Ma)"]) == 28.0
        assert float(values["Measured zircon (U-Th)/He standard deviation (Ma)"]) == 1.5


class TestCliWarning:
    @pytest.mark.parametrize(
        "details, expected",
        [
            (None, "tc1d-cli: warning: Something to note.\n"),
            (
                ["  - detail 1", "  - detail 2"],
                "tc1d-cli: warning: Something to note.\n  - detail 1\n  - detail 2\n",
            ),
        ],
    )
    def test_cli_warning(self, capsys, details, expected):
        if details is None:
            _cli_warning("Something to note.")
        else:
            _cli_warning("Something to note.", details)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == expected


class TestScreenOutput:
    def test_no_pass_number_for_single_pass(self, monkeypatch, tmp_path, capsys):
        # The pass number is only shown when the model uses more than one pass
        TestCrashFixes._run(monkeypatch, tmp_path, verbose=1, echo_info=False)
        output = capsys.readouterr().out
        assert "--- Calculating transient thermal model ---" in output
        assert "(Pass" not in output

    @pytest.mark.parametrize("echo_inputs, verbose", [(True, 1), (True, 0), (False, 1)])
    def test_echo_inputs(self, monkeypatch, tmp_path, capsys, echo_inputs, verbose):
        TestCrashFixes._run(
            monkeypatch,
            tmp_path,
            echo_inputs=echo_inputs,
            verbose=verbose,
            temp_surf=5.0,
        )
        output = capsys.readouterr().out
        if echo_inputs and verbose >= 1:
            assert "--- Model input values ---" in output
            assert "- temp_surf: 5.0" in output
            assert "- t_total: 1.0" in output
            assert "- batch_mode:" not in output
        else:
            assert "--- Model input values ---" not in output


class TestMadTraxAft:
    """AFT ages and track lengths from MadTrax."""

    @staticmethod
    def _aft_results(madtrax_aft, ketch_aft):
        # Linear cooling from 200 to 10 °C over 50 Myr
        time_history = np.linspace(0.0, myr2sec(50.0), 501)
        temp_history = np.linspace(200.0, 10.0, 501)
        depth_history = np.linspace(8000.0, 0.0, 501)
        pressure_history = depth_history * 2700.0 * 9.81
        params = init_params(madtrax_aft=madtrax_aft, ketch_aft=ketch_aft, verbose=0)
        results = calculate_ages_and_tcs(
            params,
            time_history,
            temp_history,
            depth_history,
            pressure_history,
        )
        # AFT age, mean track length, closure temperature and track-length
        # distribution
        return results[3], results[4], results[5], results[11]

    def test_madtrax_aft_age_in_ma(self):
        mt_age, mt_ftl, mt_temp, _ = self._aft_results(
            madtrax_aft=True, ketch_aft=False
        )
        ke_age, _, _, _ = self._aft_results(madtrax_aft=False, ketch_aft=True)
        # Age should be in Ma and comparable to the Ketcham age
        assert 0.0 < mt_age < 50.0
        assert mt_age == pytest.approx(ke_age, rel=0.2)
        assert 60.0 < mt_temp < 150.0
        assert 0.0 < mt_ftl < 17.0


class TestAftTrackLengths:
    """AFT track-length distributions from the Ketcham and MadTrax models."""

    @pytest.mark.parametrize("madtrax_aft", [True, False])
    def test_track_length_distribution(self, madtrax_aft):
        _, mean_ftl, _, ftl_dist = TestMadTraxAft._aft_results(
            madtrax_aft=madtrax_aft,
            ketch_aft=not madtrax_aft,
        )
        length = ftl_dist[:, 0]
        prob = ftl_dist[:, 1]
        if madtrax_aft:
            # 1-µm bins from 0-1 µm to 16-17 µm
            assert len(length) == 17
            assert length[0] == 0.5
            assert length[-1] == 16.5
        # Probability densities should integrate to one, and the mean length
        # should match the mean length from the age model
        bin_width = length[1] - length[0]
        assert prob.sum() * bin_width == pytest.approx(1.0, abs=1e-3)
        assert (length * prob).sum() * bin_width == pytest.approx(mean_ftl, abs=0.05)

    @pytest.mark.parametrize(
        "madtrax_aft, write_ft_lengths", [(True, True), (False, False)]
    )
    def test_track_length_plot(
        self, monkeypatch, tmp_path, madtrax_aft, write_ft_lengths
    ):
        # The plot should not depend on writing the track lengths to file
        monkeypatch.chdir(tmp_path)
        params = init_params(
            time=10.0,
            dt=20000.0,
            nx=21,
            ero_type=1,
            ero_option1=5.0,
            madtrax_aft=madtrax_aft,
            write_ft_lengths=write_ft_lengths,
            plot_results=True,
            plot_ft_length_dist=True,
            save_plots=True,
            display_plots=False,
            verbose=0,
        )
        prep_model(params)
        assert (tmp_path / "png" / "apatite_ft_length_dist.png").exists()
        ftl_files = [f.name for f in tmp_path.rglob("ft_length*.csv")]
        if write_ft_lengths:
            assert ftl_files == ["ft_length.csv"]
        else:
            assert ftl_files == []

    def test_track_length_files_for_past_ages(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        params = init_params(
            time=10.0,
            dt=20000.0,
            nx=21,
            ero_type=1,
            ero_option1=5.0,
            past_age_increment=5.0,
            write_ft_lengths=True,
            plot_results=False,
            verbose=0,
        )
        prep_model(params)
        files = sorted(f.name for f in tmp_path.rglob("ft_length*.csv"))
        assert files == ["ft_length_0.0Ma.csv", "ft_length_5.0Ma.csv"]


class TestAftModelSelection:
    """Only one AFT model is used at a time."""

    @pytest.mark.parametrize(
        "madtrax_aft, ketch_aft, calc_ages, expected_ketch_aft",
        [
            # MadTrax replaces Ketcham
            (True, True, True, False),
            (True, False, True, False),
            # Ketcham by default
            (False, True, True, True),
            # No AFT model is needed if ages are not calculated
            (False, False, False, False),
        ],
    )
    def test_set_aft_model(self, madtrax_aft, ketch_aft, calc_ages, expected_ketch_aft):
        params = {
            "madtrax_aft": madtrax_aft,
            "ketch_aft": ketch_aft,
            "calc_ages": calc_ages,
        }
        set_aft_model(params)
        assert params["madtrax_aft"] is madtrax_aft
        assert params["ketch_aft"] is expected_ketch_aft

    def test_no_aft_model_selected(self):
        params = {"madtrax_aft": False, "ketch_aft": False, "calc_ages": True}
        with pytest.raises(ValueError, match="No AFT age model selected"):
            set_aft_model(params)

    @staticmethod
    def _aft_output(monkeypatch, tmp_path, capsys, **kwargs):
        monkeypatch.chdir(tmp_path)
        params = init_params(
            time=10.0,
            dt=20000.0,
            nx=21,
            ero_type=1,
            ero_option1=5.0,
            plot_results=False,
            verbose=0,
            **kwargs,
        )
        prep_model(params)
        output = capsys.readouterr().out
        return [line for line in output.splitlines() if "AFT age" in line]

    def test_single_aft_age_printed(self, monkeypatch, tmp_path, capsys):
        ketcham = self._aft_output(monkeypatch, tmp_path, capsys)
        madtrax = self._aft_output(monkeypatch, tmp_path, capsys, madtrax_aft=True)
        assert len(ketcham) == 1
        assert ketcham[0].endswith("(Ketcham)")
        assert len(madtrax) == 1
        assert madtrax[0].endswith("(MadTrax)")
        # The printed age should be the MadTrax age, not the Ketcham age
        assert madtrax[0].split("(")[0] != ketcham[0].split("(")[0]

    @pytest.mark.parametrize(
        "age_prediction, ketch_aft, madtrax_aft",
        [
            ({}, True, False),
            ({"ketch_aft": True, "madtrax_aft": True}, False, True),
        ],
    )
    def test_cli_yaml_options(
        self, monkeypatch, tmp_path, age_prediction, ketch_aft, madtrax_aft
    ):
        import yaml
        import tc1d.tc1d as tc1d_module

        config = {
            "general": {"run_type": "forward"},
            "erosion_model": {"ero_type": 1, "ero_option1": 1.0},
            "age_prediction": age_prediction,
        }
        input_file = tmp_path / "input.yaml"
        input_file.write_text(yaml.safe_dump(config))

        captured = {}

        def fake_run_model(params):
            captured["params"] = params

        monkeypatch.setattr(tc1d_module, "run_model", fake_run_model)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["tc1d-cli", "--input-file", str(input_file)])
        tc1d_cli.main()
        assert captured["params"]["ketch_aft"] is ketch_aft
        assert captured["params"]["madtrax_aft"] is madtrax_aft

    def test_cli_no_aft_model(self, monkeypatch, tmp_path):
        import yaml

        config = {
            "general": {"run_type": "forward"},
            "erosion_model": {"ero_type": 1, "ero_option1": 1.0},
            "age_prediction": {"ketch_aft": False},
        }
        input_file = tmp_path / "input.yaml"
        input_file.write_text(yaml.safe_dump(config))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["tc1d-cli", "--input-file", str(input_file)])
        with pytest.raises(ValueError, match="No AFT age model selected"):
            tc1d_cli.main()


class TestMadTraxPadTime:
    """Time padding is included in MadTrax AFT and ZFT ages."""

    @staticmethod
    def _ages(temp_start, pad_time):
        # Linear cooling from temp_start to 10 °C over 20 Myr
        time_history = np.linspace(0.0, myr2sec(20.0), 201)
        temp_history = np.linspace(temp_start, 10.0, 201)
        depth_history = np.linspace(5000.0, 0.0, 201)
        pressure_history = depth_history * 2700.0 * 9.81
        params = init_params(
            madtrax_aft=True, ketch_aft=False, pad_time=pad_time, verbose=0
        )
        results = calculate_ages_and_tcs(
            params,
            time_history,
            temp_history,
            depth_history,
            pressure_history,
        )
        # MadTrax AFT and ZFT ages
        return results[3], results[9]

    def test_padding_makes_unreset_ages_older(self):
        # Neither age is reset when cooling from 40 °C, so the padding time
        # should be added to both ages
        aft_age, zft_age = self._ages(temp_start=40.0, pad_time=0.0)
        aft_age_pad, zft_age_pad = self._ages(temp_start=40.0, pad_time=50.0)
        assert aft_age_pad > aft_age + 40.0
        assert zft_age_pad > zft_age + 40.0

    def test_padding_does_not_change_reset_ages(self):
        # Both ages are reset when cooling from 400 °C. MadTrax uses at least
        # 100 time steps, so the ages differ slightly as the padding makes the
        # time steps longer.
        aft_age, zft_age = self._ages(temp_start=400.0, pad_time=0.0)
        aft_age_pad, zft_age_pad = self._ages(temp_start=400.0, pad_time=50.0)
        assert aft_age_pad == pytest.approx(aft_age, abs=0.5)
        assert zft_age_pad == pytest.approx(zft_age, abs=0.5)


class TestTimePadding:
    """Time padding of thermal histories."""

    @pytest.mark.parametrize(
        "pad_time, expected", [(0.0, []), (3.0, [11.0, 12.0, 13.0])]
    )
    def test_pad_times(self, pad_time, expected):
        time_ma = np.array([10.0, 5.0, 0.0])
        pad_times = get_pad_times({"pad_time": pad_time}, time_ma)
        assert len(pad_times) == len(expected)
        assert np.allclose(pad_times, expected)

    def test_tt_path_with_padding(self):
        params = init_params(pad_time=2.0)
        time_ma = np.linspace(4.0, 0.0, 5)
        temp_history = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
        time_path, temp_path = get_tt_path(params, time_ma, temp_history)
        # Youngest point first, then the oldest point in the history and the
        # padding points at the temperature of the oldest point
        assert time_path[0] == 0.0
        assert temp_path[0] == 20.0
        assert np.allclose(time_path[-3:], [4.0, 5.0, 6.0])
        assert np.allclose(temp_path[-3:], [100.0, 100.0, 100.0])

    def test_ttdp_history_with_padding(self, tmp_path):
        params = init_params(pad_time=2.0)
        time_history = np.array([0.0, myr2sec(1.0), myr2sec(2.0)])
        temp_history = np.array([100.0, 60.0, 20.0])
        depth_history = np.array([3000.0, 2000.0, 1000.0])
        pressure_history = np.array([80.0e6, 50.0e6, 30.0e6])
        ttdp_file = tmp_path / "ttdp.csv"
        save_ttdp_history(
            params,
            ttdp_file,
            time_history,
            temp_history,
            depth_history,
            pressure_history,
        )
        ttdp = np.genfromtxt(ttdp_file, delimiter=",", skip_header=1)
        # Youngest point first, then the oldest point in the history and the
        # padding points with the values of the oldest point
        assert np.allclose(ttdp[0], [0.0, 20.0, 1000.0, 30.0])
        assert np.allclose(ttdp[-3:, 0], [2.0, 3.0, 4.0])
        for row in ttdp[-3:]:
            assert np.allclose(row[1:], [100.0, 3000.0, 80.0])


class TestInverseOutput:
    """Output files for individual models are not written for inversions."""

    OUTPUT = {
        "write_tt_history": True,
        "write_ttdp_history": True,
        "write_ft_lengths": True,
        "write_temps": True,
        "write_past_ages": True,
        "write_age_output": True,
    }

    def test_disable_output_for_inversions(self):
        params = {"inverse_mode": True, **self.OUTPUT, "write_temps": False}
        disabled = disable_output_for_inversions(params)
        assert "write_temps" not in disabled
        assert len(disabled) == len(self.OUTPUT) - 1
        for key in self.OUTPUT:
            assert params[key] is False

    def test_output_kept_for_other_runs(self):
        params = {"inverse_mode": False, **self.OUTPUT}
        assert disable_output_for_inversions(params) == []
        for key in self.OUTPUT:
            assert params[key] is True

    def test_cli_inversion_disables_output(self, monkeypatch, tmp_path):
        with pytest.warns(UserWarning, match="not written for NA or MCMC"):
            params = TestRunModeFlags._run_cli(
                monkeypatch,
                tmp_path,
                "na",
                {"ero_type": 1, "ero_option1": [1.0, 5.0]},
                output=self.OUTPUT,
            )
        for key in self.OUTPUT:
            assert params[key] is False
