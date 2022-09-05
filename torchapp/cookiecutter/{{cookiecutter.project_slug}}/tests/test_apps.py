from torchapp.testing import TorchAppTestCase
from {{ cookiecutter.project_slug }}.apps import {{ cookiecutter.app_name }}


class Test{{ cookiecutter.app_name }}(TorchAppTestCase):
    app_class = {{ cookiecutter.app_name }}
