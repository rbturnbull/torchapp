from pathlib import Path
import typer

from cookiecutter.main import cookiecutter as cookiecutter_main

app = typer.Typer()


@app.command()
def cookiecutter(
    template: str = typer.Option(
        "",
        help="The Cookiecutter template to use. "
        "If left blank then it uses the default template in this version of torchapp. "
        "If 'gh' or 'github' then it uses the latest torchapp cookiecutter template at https://github.com/rbturnbull/torchapp-cookiecutter",
    ),
    checkout: str = typer.Option(
        None, help="A branch, tag or commit to checkout after git clone"
    ),
    no_input: bool = typer.Option(
        False, help="Do not prompt for parameters and only use cookiecutter.json file content"
    ),
    extra_context: str = typer.Option(None, help=""),
    replay: bool = typer.Option(
        False, help="Do not prompt for parameters and only use information entered previously"
    ),
    overwrite_if_exists: bool = typer.Option(
        False, help="Overwrite the contents of the output directory if it already exists"
    ),
    output_dir: Path = typer.Option(
        ".", help="Where to output the generated project dir into"
    ),
    config_file: Path = typer.Option(None, help="User configuration file"),
    default_config: bool = typer.Option(
        False, help="Do not load a config file. Use the defaults instead"
    ),
    directory: str = typer.Option(
        None,
        help="Directory within repo that holds cookiecutter.json file for advanced repositories with multi templates in it",
    ),
    skip_if_file_exists: bool = typer.Option(
        False, help="Skip the files in the corresponding directories if they already exist"
    ),
):
    """
    Generate a torchapp project using Cookiecutter (https://github.com/audreyr/cookiecutter)
    """
    if not template:
        template = Path(__file__).parent.resolve() / "cookiecutter"
    elif template in ["gh", "github"]:
        template = "https://github.com/rbturnbull/torchapp-cookiecutter"

    return cookiecutter_main(
        template=str(template),
        checkout=checkout,
        no_input=no_input,
        extra_context=extra_context,
        replay=replay,
        overwrite_if_exists=overwrite_if_exists,
        output_dir=output_dir,
        config_file=config_file,
        default_config=default_config,
        directory=directory,
        skip_if_file_exists=skip_if_file_exists,
    )
