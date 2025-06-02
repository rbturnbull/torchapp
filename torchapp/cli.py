import sys
import types
from typing import Any, Callable
from dataclasses import dataclass
import copy
from inspect import signature, Parameter
import click
import typer
from typer.models import OptionInfo
from typer.core import TyperCommand
from rich.console import Console

console = Console()

def make_flag_function(attr: Callable) -> Callable:
    def run_flag_function(ctx, param, value):
        if value:
            result = attr()
            if result is not None:
                console.print(result.strip())
            raise typer.Exit()
    return run_flag_function


class CLIAppTyper(typer.Typer):
    def __init__(
        self,
        cliapp: "CLIApp",
        pretty_exceptions_enable: bool = False,
        add_completion:bool=False,
        **kwargs: Any
    ):
        super().__init__(pretty_exceptions_enable=pretty_exceptions_enable, add_completion=add_completion, **kwargs)
        self.cliapp = cliapp

    def getcommand(self) -> click.Command:
        command = typer.main.get_command(self)
        self.patch_command(command)
        return command

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if sys.excepthook != typer.main.except_hook:
            sys.excepthook = typer.main.except_hook
        try:
            command = self.getcommand()
            return command(*args, **kwargs)
        except Exception as e:
            # Set a custom attribute to tell the hook to show nice exceptions for user
            # code. An alternative/first implementation was a custom exception with
            # raise custom_exc from e
            # but that means the last error shown is the custom exception, not the
            # actual error. This trick improves developer experience by showing the
            # actual error last.
            setattr(
                e,
                typer.main._typer_developer_exception_attr_name,
                typer.main.DeveloperExceptionConfig(
                    pretty_exceptions_enable=self.pretty_exceptions_enable,
                    pretty_exceptions_show_locals=self.pretty_exceptions_show_locals,
                    pretty_exceptions_short=self.pretty_exceptions_short,
                ),
            )
            raise e

    def patch_command(self, cmd: click.Command) -> click.Command:
        flag_index = 0
        for attr_name in dir(self.cliapp):
            attr = getattr(self.cliapp, attr_name)

            if not isinstance(attr, Method):
                continue

            if not attr.flag:
                continue

            # Add shortcut and flag to the command
            option_params = [f"--{attr_name}"]
            if attr.shortcut:
                shortcut = attr.shortcut
                if shortcut[0] != "-":
                    shortcut = f"-{shortcut}"
                option_params.append(shortcut)
                
            cmd.params.insert(
                flag_index,
                click.Option(
                    option_params,
                    is_flag=True,
                    is_eager=True,
                    expose_value=False,
                    help=(attr.__doc__ or "").strip().splitlines()[0] if attr.__doc__ else None,
                    callback=make_flag_function(attr),
                )
            )
            flag_index += 1


# def launch_gui(typer_app:typer.Typer, share:bool=False):
#     from guigaga.guigaga import GUIGAGA   
#     gui = GUIGAGA(
#         typer.main.get_group(typer_app), 
#         # click_context=ctx,
#         # theme=theme,
#         allow_file_download=False,
#     )
#     gui.launch(launch_kwargs={"share": share})    


@dataclass
class Method():
    func: Callable
    methods_to_call: list[str]
    main: bool = False    
    tool: bool = False    
    flag: bool = False
    shortcut: str = ""
    signature_ready: bool = False
    obj = None

    def __post_init__(self):
        self.__code__ = self.func.__code__
        self.__defaults__ = self.func.__defaults__

    @property
    def __name__(self):
        return self.func.__name__
    
    def get_kwargs(self, kwargs):
        parameters = signature(self.func).parameters
        func_kwargs = {k: v for k, v in kwargs.items() if k in parameters}
        original_kwargs = func_kwargs.copy()
        
        # Replace default values if they are OptionInfo
        for key,value in parameters.items(): 
            if key not in func_kwargs and isinstance(value.default, OptionInfo):
                func_kwargs[key] = value.default.default

        if not hasattr(self.obj, 'original_kwargs'):
            self.obj.original_kwargs = dict()
        self.obj.original_kwargs[self.__name__] = kwargs['opts'] if 'opts' in kwargs else original_kwargs
        return func_kwargs

    def __call__(self, *args, **kwargs):
        func_kwargs = self.get_kwargs(kwargs)
        return self.func(self.obj, *args, **func_kwargs)

    @property
    def __signature__(self):
        return signature(self.func)

    @property
    def __doc__(self):
        return self.func.__doc__


def method(*args, main:bool=False, tool:bool=False, flag:bool=False, shortcut:str=""):
    if len(args) == 1 and callable(args[0]):
        return Method(args[0], [], main=main, tool=tool, flag=flag, shortcut=shortcut)
    
    def decorator(func):
        return Method(func, args, main=main, tool=tool, flag=flag, shortcut=shortcut)

    return decorator


def tool(*methods_to_call, **kwargs):
    return method(*methods_to_call, tool=True, **kwargs)


def main(*methods_to_call, **kwargs):
    return method(*methods_to_call, main=True, tool=True, **kwargs)


def flag(*methods_to_call, **kwargs):
    return method(*methods_to_call, flag=True, **kwargs)


def collect_arguments(*funcs):
    """Collect arguments from multiple functions."""
    params = {}
    for func in funcs:
        for name, param in signature(func).parameters.items():
            # if param.default != Parameter.empty:
            #     # only include args with defaults
            #     continue
            if name != "self":  # Exclude 'self' parameter
                params[name] = param
    return params


class CLICommand(TyperCommand):
    def parse_args(self, ctx, args):
        original_args = list(args)
        result = super().parse_args(ctx, args)

        # Save Options given
        parser = self.make_parser(ctx)
        opts, _, _ = parser.parse_args(args=original_args)
        ctx.params['opts'] = opts
        
        return result


def clone_function(f):
    new_func = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__
    )
    new_func.__annotations__ = dict(f.__annotations__)  # Copy type hints
    return new_func


class CLIApp:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main_app = CLIAppTyper(self)
        self.tools_app = CLIAppTyper(self)
        self.register_methods()

    @classmethod
    def main(cls):
        main_app = cls().main_app
        main_app()

    @classmethod
    def tools(cls):
        cls().tools_app()
        
    # @tool
    # def gui(self, share:bool=False):
    #     """ Launches a GUI for the tool commands. """
    #     launch_gui(self.tools_app, share=share)

    def add_to_app(self, app:typer.Typer, func:Method) -> Method:
        if not func.flag:
            app.command(cls=CLICommand)(func)
        return func

    def add_to_main(self, func:Method) -> Method:
        return self.add_to_app(self.main_app, func)

    def add_to_tools(self, func:Method) -> Method:
        return self.add_to_app(self.tools_app, func)

    def register_methods(self):
        methods = []
        # Make copy of the method to avoid modifying the original
        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            if not isinstance(attr, Method):
                continue

            method = copy.deepcopy(attr)
            method.func = clone_function(attr.func)
            setattr(self, attr_name, method)
            methods.append(method)

        for method in methods:
            # Add to the CLI if method is decorated as a command
            if method.main:
                self.add_to_main(method)
            if method.tool:
                self.add_to_tools(method)

            # Modify the signature of the method if necessary
            if not method.signature_ready:
                self.modify_signature(method)

    def modify_signature(self, method_to_modify:Method, **kwargs) -> None:
        # Check if the method is already had its signature modified
        if not isinstance(method_to_modify, Method) or method_to_modify.signature_ready:
            return
                    
        # if "tune" == method_to_modify.__name__:
        #     x = collect_arguments(self.tune)
        #     breakpoint()


        method_to_modify.obj = self

        all_methods = [method_to_modify]
        for method_to_call_name in method_to_modify.methods_to_call:
            if method_to_call_name == "super":
                self_super = super(self.__class__, self) # this 'self' is a problem. TODO add an object as a function argument
                method_to_call = getattr(self_super, method_to_modify.__name__)
            else:
                method_to_call = getattr(self, method_to_call_name)

            # make sure method is has its signature modified before getting parameters
            self.modify_signature(method_to_call)
            all_methods.append(method_to_call)

        # Get all arguments from all methods
        params = collect_arguments(*all_methods)
        new_params = [
            Parameter(name, param.kind, default=param.default, annotation=param.annotation)
            for name, param in params.items()
            if name not in ["self", "kwargs"] and param.default != Parameter.empty
        ]

        # if "tune" == method_to_modify.__name__:
        #     breakpoint()

        try:
            method_to_modify.func.__signature__ = signature(method_to_modify.func).replace(parameters=new_params)
        except Exception as err:
            print(f"ERROR in {method_to_modify}: {err}")
        
        # Set the method as ready
        method_to_modify.signature_ready = True
