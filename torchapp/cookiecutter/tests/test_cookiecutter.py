# Based on https://github.com/simonw/datasette-plugin/blob/main/tests/test_cookiecutter_template.py
from cookiecutter.main import cookiecutter
from pathlib import Path


def test_cookiecutter(tmpdir):
    generate(
        tmpdir,
        {
            "project_name": "Test App",
        },
    )
    paths = get_paths(tmpdir)
    assert paths == {
        "test_app/LICENSE",
        "test_app/.github/ISSUE_TEMPLATE/feature_request.md",
        "test_app/docs/conf.py",
        "test_app/.github/workflows",
        "test_app/docs/api.rst",
        "test_app/docs/credits.rst",
        "test_app/CONTRIBUTING.rst",
        "test_app/.github/ISSUE_TEMPLATE",
        "test_app/.gitignore",
        "test_app/.github/workflows/docs.yml",
        "test_app/tests",
        "test_app/CODE_OF_CONDUCT.md",
        "test_app/docs/Makefile",
        "test_app/.pre-commit-config.yaml",
        "test_app/mkdocs.sh",
        "test_app",
        "test_app/test_app",
        "test_app/docs/cli.rst",
        "test_app/.gitlab-ci.yml",
        "test_app/README.rst",
        "test_app/docs",
        "test_app/.github/workflows/publish.yml",
        "test_app/docs/index.rst",
        "test_app/docs/make.bat",
        "test_app/tests/conftest.py",
        "test_app/.coveragerc",
        "test_app/tests/test_apps.py",
        "test_app/tests/__init__.py",
        "test_app/.github",
        "test_app/docs/.gitignore",
        "test_app/test_app/__init__.py",
        "test_app/.github/workflows/testing.yml",
        "test_app/.github/ISSUE_TEMPLATE/bug_report.md",
        "test_app/test_app/apps.py",
        "test_app/pyproject.toml",
    }


def generate(directory, context):
    TEMPLATE_DIRECTORY = str(Path(__file__).parent.parent)
    cookiecutter(
        template=TEMPLATE_DIRECTORY,
        output_dir=str(directory),
        no_input=True,
        extra_context=context,
    )


def get_paths(directory):
    paths = list(Path(directory).glob("**/*"))
    paths = [r.relative_to(directory) for r in paths]
    return {str(f) for f in paths if str(f) != "."}
