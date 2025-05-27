import pytest
from torchapp.cli import CLIApp, method


class BaseCLIApp(CLIApp):
    @method
    def add_one(self, value:int) -> int:
        raise NotImplementedError


class ChildCLIApp(BaseCLIApp):
    @method
    def add_one(self, value:int):
        return value + 1


def test_multiple_instances():
    # Check that a base app can raise a not implemented error
        
    app = BaseCLIApp()
    with pytest.raises(NotImplementedError):
        app.add_one(42)

    # Check if new app can override the function

    child_app = ChildCLIApp()
    assert child_app.add_one(42) == 43

    # Check that the first app still raises not implemented
    with pytest.raises(NotImplementedError):
        app.add_one(42)

    # Check that new definition of child hasn't affected parent class
    new_base_app = BaseCLIApp()
    with pytest.raises(NotImplementedError):
        new_base_app.add_one(42)
