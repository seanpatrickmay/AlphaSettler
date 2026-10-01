from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_only_oracle_imports_catanatron():
    offenders = []
    for path in [*ROOT.glob("alphasettler/**/*.py"), *ROOT.glob("tests/python/*.py")]:
        if path.name.startswith("test_oracle"):
            continue
        text = path.read_text()
        if "import catanatron" in text or "from catanatron" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_oracle_package_init_does_not_import_catanatron():
    text = (ROOT / "oracle" / "__init__.py").read_text()
    assert "catanatron import" not in text and "import catanatron" not in text
