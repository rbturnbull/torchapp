import types
import inspect
from collections.abc import Iterable
import typer
from typer.models import OptionInfo
from typer.main import get_params_convertors_ctx_param_name_from_function
from rich.console import Console

console = Console()


def copy_func(f, name=None):
    """
    Returns a deep copy of a function.

    The new function has the same code, globals, defaults, closure, annotations, and name
    (unless a new name is provided)

    Derived from https://stackoverflow.com/a/30714299
    """
    fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__)
    fn.__annotations__.update(f.__annotations__)
    return fn


def call_func(func, *args, **kwargs):
    """
    Calls a function while filtering the kwargs for only ones in the signature.

    Args:
        func (Callable): The function to call

    Returns:
        The result of the function call.
    """
    allowed_params, _, _ = get_params_convertors_ctx_param_name_from_function(func)
    allowed_param_names = [p.name for p in allowed_params]
    kwargs = {key: value for key, value in kwargs.items() if key in allowed_param_names}
    return func(*args, **kwargs)


def change_typer_to_defaults(func):
    func = getattr(func, "__func__", func)
    signature = inspect.signature(func)

    # Create a dictionary with both the existing parameters for the function and the new ones
    parameters = dict(signature.parameters)

    for key, value in parameters.items():
        if isinstance(value.default, OptionInfo):
            parameters[key] = value.replace(default=value.default.default)

    func.__signature__ = signature.replace(parameters=parameters.values())

    # Change defaults directly
    if func.__defaults__ is not None:
        func.__defaults__ = tuple(
            [value.default if isinstance(value, OptionInfo) else value for value in func.__defaults__]
        )


def add_kwargs(to_func, from_funcs):
    """Adds all the keyword arguments from one function to the signature of another function.

    Args:
        from_funcs (callable or iterable): The function with new parameters to add.
        to_func (callable): The function which will receive the new parameters in its signature.
    """

    if not isinstance(from_funcs, Iterable):
        from_funcs = [from_funcs]

    for from_func in from_funcs:
        # Get the existing parameters
        to_func = getattr(to_func, "__func__", to_func)
        from_func = getattr(from_func, "__func__", from_func)
        from_func_signature = inspect.signature(from_func)
        to_func_signature = inspect.signature(to_func)

        # Create a dictionary with both the existing parameters for the function and the new ones
        to_func_parameters = dict(to_func_signature.parameters)

        if "kwargs" in to_func_parameters:
            kwargs_parameter = to_func_parameters.pop("kwargs")

        from_func_kwargs = {
            k: v
            for k, v in from_func_signature.parameters.items()
            if v.default != inspect.Parameter.empty and k not in to_func_parameters
        }
        # to_func_parameters['kwargs'] = kwargs_parameter

        to_func_parameters.update(from_func_kwargs)

        # Modify function signature with the parameters in this dictionary
        # print('to_func', hex(id(to_func)))
        to_func.__signature__ = to_func_signature.replace(parameters=to_func_parameters.values())
        for key, value in from_func.__annotations__.items():
            if key not in to_func.__annotations__:
                to_func.__annotations__[key] = value
