"""Published extras must not depend on source-only Git or URL requirements."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement


def test_published_dependencies_have_no_direct_urls():
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    requirements = list(project["dependencies"])
    for dependencies in project["optional-dependencies"].values():
        requirements.extend(dependencies)
    assert all(Requirement(value).url is None for value in requirements)
    source_tools = metadata["dependency-groups"]["vmm-harness"]
    assert any(Requirement(value).name == "diffusers-mm" for value in source_tools)
