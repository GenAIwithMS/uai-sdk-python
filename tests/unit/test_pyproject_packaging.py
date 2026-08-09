from pathlib import Path


def test_poetry_metadata_not_nested_under_scripts():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    lines = pyproject.read_text(encoding="utf-8").splitlines()

    scripts_idx = lines.index("[tool.poetry.scripts]")
    deps_idx = lines.index("[tool.poetry.dependencies]")

    scripts_block = [line.strip() for line in lines[scripts_idx + 1 : deps_idx] if line.strip()]
    assert scripts_block == ['uai = "uai.cli:main"']

    keywords_idx = next(i for i, line in enumerate(lines) if line.startswith("keywords = ["))
    repository_idx = next(i for i, line in enumerate(lines) if line.startswith("repository = "))
    documentation_idx = next(
        i for i, line in enumerate(lines) if line.startswith("documentation = ")
    )
    classifiers_idx = next(i for i, line in enumerate(lines) if line.startswith("classifiers = ["))

    assert keywords_idx < scripts_idx
    assert repository_idx < scripts_idx
    assert documentation_idx < scripts_idx
    assert classifiers_idx < scripts_idx
