import io
import zipfile

import pytest

from tc1d import notebooks


def make_archive(path, top="tc1d-notebooks"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{top}/explore_tc1d.ipynb", "{}")
        zf.writestr(f"{top}/data/sample_data.csv", "a,b\n1,2\n")
    return path


def test_archive_url_versions():
    base = "https://github.com/HUGG/Tc1D/releases"
    assert (
        notebooks.archive_url("0.7.0") == f"{base}/download/v0.7.0/tc1d-notebooks.zip"
    )
    assert (
        notebooks.archive_url("v0.7.0") == f"{base}/download/v0.7.0/tc1d-notebooks.zip"
    )
    assert (
        notebooks.archive_url("latest") == f"{base}/latest/download/tc1d-notebooks.zip"
    )


@pytest.mark.parametrize(
    "ver, expected",
    [("0.7.0", True), ("dev", False), ("0.7.1.dev3", False), ("0.7.0+g1234", False)],
)
def test_is_release_version(ver, expected):
    assert notebooks.is_release_version(ver) is expected


def test_extract_archive_strips_top_level_dir(tmp_path):
    archive = make_archive(tmp_path / "nb.zip")
    dest = tmp_path / "out"
    notebooks.extract_archive(archive, dest)
    assert (dest / "explore_tc1d.ipynb").is_file()
    assert (dest / "data" / "sample_data.csv").is_file()


def test_extract_archive_rejects_unsafe_paths(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        notebooks.extract_archive(archive, tmp_path / "out")


@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    data = make_archive(tmp_path / "remote.zip").read_bytes()
    urls = []

    def fake_urlopen(url):
        urls.append(url)
        return io.BytesIO(data)

    monkeypatch.setattr(notebooks.urllib.request, "urlopen", fake_urlopen)
    return urls


def test_fetch_notebooks(tmp_path, fake_download):
    dest = notebooks.fetch_notebooks(dest=tmp_path / "nb", ver="0.7.0")
    assert (dest / "explore_tc1d.ipynb").is_file()
    assert fake_download == [notebooks.archive_url("0.7.0")]


def test_fetch_notebooks_existing_dest(tmp_path, fake_download):
    dest = tmp_path / "nb"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    with pytest.raises(FileExistsError):
        notebooks.fetch_notebooks(dest=dest, ver="0.7.0")
    assert not fake_download
    notebooks.fetch_notebooks(dest=dest, ver="0.7.0", force=True)
    assert not (dest / "old.txt").exists()
    assert (dest / "explore_tc1d.ipynb").is_file()


def test_fetch_notebooks_dev_version_uses_latest(tmp_path, fake_download, monkeypatch):
    monkeypatch.setattr(notebooks, "__version__", "0.7.1.dev2")
    notebooks.fetch_notebooks(dest=tmp_path / "nb")
    assert fake_download == [notebooks.archive_url("latest")]
