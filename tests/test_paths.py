from small_lm_lab.paths import BULK_ROOT_ENV, REPO_ROOT, portable_path


def test_portable_path_rewrites_bulk_artifacts() -> None:
    path = REPO_ROOT.parent / "outside"
    assert portable_path(path) == str(path)


def test_portable_path_keeps_repository_artifacts_relative() -> None:
    assert portable_path(REPO_ROOT / "analysis" / "result.json") == "analysis/result.json"


def test_portable_path_names_bulk_environment() -> None:
    from small_lm_lab import paths

    artifact = paths.BULK_ROOT / "checkpoints" / "model.pt"
    assert portable_path(artifact) == f"${BULK_ROOT_ENV}/checkpoints/model.pt"
