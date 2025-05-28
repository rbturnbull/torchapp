import sys
from typing import Callable
import click
from typing import Any
import typer
from typer.core import TyperCommand
from inspect import signature, Parameter
from typer.models import OptionInfo
from dataclasses import dataclass
import guigaga
from guigaga.interface import InterfaceBuilder
from guigaga.themes import Theme
from typing import Callable, Optional


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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if sys.excepthook != typer.main.except_hook:
            sys.excepthook = typer.main.except_hook
        try:
            command = typer.main.get_command(self)
            self.patch_command(command)
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
        global_option_index = 0
        for attr_name in dir(self.cliapp):
            attr = getattr(self.cliapp, attr_name)

            if not isinstance(attr, Method):
                continue

            if not attr.global_option:
                continue

            def run_option(ctx, param, value):
                if value:
                    result = attr()
                    if result is not None:
                        typer.echo(result)
                    raise typer.Exit()

            cmd.params.insert(
                global_option_index,
                click.Option(
                    ["--version", "-v"],
                    is_flag=True,
                    is_eager=True,
                    expose_value=False,
                    help="Show version and exit",
                    callback=run_option,
                )
            )
            global_option_index += 1


def launch_gui(typer_app:typer.Typer):
    app = typer.main.get_command(typer_app)

    def update_launch_kwargs_from_cli(ctx, launch_kwargs, cli_mappings):
        """
        Update launch_kwargs with CLI options that differ from their defaults.

        Args:
            ctx: Click context object containing the command parameters and options.
            launch_kwargs: Dictionary to update with CLI-specified values.
            cli_mappings: Dictionary mapping CLI option names to their corresponding launch_kwargs keys.
        """
        for param in ctx.command.params:
            param_name = param.name
            if param_name in cli_mappings and ctx.params[param_name] != param.default:
                launch_kwargs[cli_mappings[param_name]] = ctx.params[param_name]

    name: Optional[str] = None
    command_name: str = "gui"
    message: str = "Open Gradio GUI."
    theme: Theme = Theme.base
    hide_not_required: bool = False
    allow_file_download: bool = False
    launch_kwargs: Optional[dict] = {}
    queue_kwargs: Optional[dict] = {}
    catch_errors: bool = True

    ctx = app.make_context("info_name", args=[])

    # Mapping of CLI option names to launch_kwargs keys
    cli_mappings = {
        "share": "share",
        "host": "server_name",
        "port": "server_port",
    }

    # Update launch_kwargs based on CLI inputs
    update_launch_kwargs_from_cli(ctx, launch_kwargs, cli_mappings)

    # Build the interface using InterfaceBuilder
    builder = InterfaceBuilder(
        app,
        app_name=name,
        command_name=command_name,
        # click_context=ctx,
        theme=theme,
        hide_not_required=hide_not_required,
        allow_file_download=allow_file_download,
        # catch_errors=catch_errors,
    )

    # Launch the interface with optional sharing
    builder.interface.queue(**queue_kwargs).launch(**launch_kwargs, app_kwargs={"docs_url": "/docs"})


@dataclass
class Method():
    func: Callable
    methods_to_call: list[str]
    main: bool = False    
    tool: bool = False    
    global_option: bool = False
    signature_ready: bool = False
    obj = None

    @property
    def __name__(self):
        return self.func.__name__

    def __call__(self, *args, **kwargs):
        parameters = signature(self.func).parameters
        func_kwargs = {k: v for k, v in kwargs.items() if k in parameters}
        
        # Replace default values if they are OptionInfo
        for key,value in parameters.items(): 
            if key not in func_kwargs and isinstance(value.default, OptionInfo):
                func_kwargs[key] = value.default.default

        if 'opts' in kwargs:
            self.obj.opts = kwargs['opts']
        return self.func(self.obj, *args, **func_kwargs)

    @property
    def __signature__(self):
        return signature(self.func)

    @property
    def __doc__(self):
        return self.func.__doc__


def method(*args, main:bool=False, tool:bool=False, global_option:bool=False):
    if len(args) == 1 and callable(args[0]):
        return Method(args[0], [], main=main, tool=tool, global_option=global_option)
    
    def decorator(func):
        return Method(func, args, main=main, tool=tool, global_option=global_option)

    return decorator


def tool(*methods_to_call, **kwargs):
    return method(*methods_to_call, tool=True, **kwargs)


def main(*methods_to_call, **kwargs):
    return method(*methods_to_call, main=True, tool=True, **kwargs)


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

    @classmethod
    def tools_gui(cls):
        launch_gui(cls().tools_app)
        
    @classmethod
    def main_gui(cls):
        launch_gui(cls().main_app)
        
    @tool
    def gui(self):
        """ Launches a GUI for the tool commands. """
        launch_gui(self.tools_app)

    def add_to_app(self, app:typer.Typer, func:Method) -> Method:
        if not func.global_option:
            app.command(cls=CLICommand)(func)
        return func

    def add_to_main(self, func:Method) -> Method:
        return self.add_to_app(self.main_app, func)

    def add_to_tools(self, func:Method) -> Method:
        return self.add_to_app(self.tools_app, func)

    def register_methods(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            if not isinstance(attr, Method):
                continue

            # Add to the CLI if method is decorated as a command
            if attr.main:
                self.add_to_main(attr)
            if attr.tool:
                self.add_to_tools(attr)

            # Modify the signature of the method if necessary
            if not attr.signature_ready:
                self.modify_signature(attr)

    def modify_signature(self, method_to_modify:Method, **kwargs) -> None:
        # Check if the method is already had its signature modified
        if not isinstance(method_to_modify, Method) or method_to_modify.signature_ready:
            return
        
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

        try:
            method_to_modify.func.__signature__ = signature(method_to_modify.func).replace(parameters=new_params)
        except Exception as err:
            print(f"ERROR in {method_to_modify}: {err}")
        
        # Set the method as ready
        method_to_modify.signature_ready = True
