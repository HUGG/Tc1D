#!/usr/bin/env python3
"""Download the Tc1D example notebooks and data.

The notebooks are not included in the Python package itself. Instead, a ZIP
archive of the ``notebooks`` directory is attached to each GitHub release of
Tc1D. This module downloads the archive matching the installed version of
Tc1D (or a version given by the user) and extracts it locally.
"""

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    from importlib.metadata import version

    __version__ = version("tc1d")
except ImportError:
    __version__ = "dev"

REPO_URL = "https://github.com/HUGG/Tc1D"
ASSET_NAME = "tc1d-notebooks.zip"
DEFAULT_DEST = "tc1d-notebooks"


def is_release_version(ver):
    """Check whether a version string looks like a published release."""
    return ver not in ("", "dev", "latest") and not any(
        tag in ver for tag in ("dev", "+")
    )


def archive_url(ver="latest"):
    """Return the download URL for the notebook archive of a given release.

    Parameters
    ----------
    ver : str
        Tc1D version (e.g., ``"0.7.0"`` or ``"v0.7.0"``), or ``"latest"`` for
        the most recent release.

    Returns
    -------
    str
        URL of the notebook archive.
    """
    if ver == "latest":
        return f"{REPO_URL}/releases/latest/download/{ASSET_NAME}"
    tag = ver if ver.startswith("v") else f"v{ver}"
    return f"{REPO_URL}/releases/download/{tag}/{ASSET_NAME}"


def extract_archive(archive, dest):
    """Extract a notebook archive into ``dest``.

    The archive is expected to contain a single top-level directory, whose
    contents are placed directly into ``dest``.
    """
    dest = Path(dest)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            # Refuse members that would end up outside the extraction folder
            for name in zf.namelist():
                target = (tmp / name).resolve()
                if not target.is_relative_to(tmp.resolve()):
                    raise ValueError(f"Unsafe path in archive: {name}")
            zf.extractall(tmp)
        entries = list(tmp.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
        shutil.copytree(root, dest)


def fetch_notebooks(dest=DEFAULT_DEST, ver=None, force=False):
    """Download and extract the Tc1D notebooks.

    Parameters
    ----------
    dest : str or Path
        Directory where the notebooks will be placed.
    ver : str or None
        Version of the notebooks to download. Defaults to the installed
        version of Tc1D, or the latest release for development installs.
    force : bool
        Replace ``dest`` if it already exists.

    Returns
    -------
    Path
        Path to the directory containing the notebooks.
    """
    dest = Path(dest)
    if ver is None:
        if is_release_version(__version__):
            ver = __version__
        else:
            print(
                f"Installed Tc1D version ({__version__}) is not a release, "
                "downloading notebooks from the latest release."
            )
            ver = "latest"

    if dest.exists() and not force:
        raise FileExistsError(
            f"'{dest}' already exists. Use --force to replace it or "
            "--dest to choose another location."
        )

    url = archive_url(ver)
    print(f"Downloading Tc1D notebooks ({ver}) from {url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / ASSET_NAME
        try:
            with urllib.request.urlopen(url) as response, open(archive, "wb") as f:
                shutil.copyfileobj(response, f)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise RuntimeError(
                    f"No notebook archive found for version '{ver}'. Releases "
                    "older than the introduction of this command do not include "
                    "one. Try '--version latest', or download the notebooks "
                    f"from {REPO_URL}/tree/main/notebooks"
                ) from err
            raise
        # Extract to a temporary location first so an existing destination is
        # only replaced once the download has succeeded
        extracted = Path(tmp) / "extracted"
        extract_archive(archive, extracted)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(extracted, dest)

    print(f"Notebooks saved to {dest.resolve()}")
    return dest


def main(argv=None):
    """Command-line entry point for ``tc1d-notebooks``."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the Tc1D example Jupyter notebooks and data. By default "
            "the notebooks matching the installed Tc1D version are downloaded."
        )
    )
    parser.add_argument(
        "-d",
        "--dest",
        default=DEFAULT_DEST,
        help=f"Directory to save the notebooks in (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "-v",
        "--version",
        dest="ver",
        metavar="VERSION",
        default=None,
        help=(
            "Tc1D version of the notebooks to download, e.g. 0.7.0, or 'latest' "
            f"(default: installed version, {__version__})"
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Replace the destination directory if it already exists",
    )
    args = parser.parse_args(argv)

    try:
        fetch_notebooks(dest=args.dest, ver=args.ver, force=args.force)
    except (FileExistsError, RuntimeError, urllib.error.URLError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
