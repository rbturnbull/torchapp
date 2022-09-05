from fastai.callback.core import Callback
import mlflow
import matplotlib
import pandas as pd
import pickle
from pathlib import Path
from typing import Union, Optional
from rich.console import Console

console = Console()


def get_or_create_experiment(experiment_name: str) -> mlflow.entities.Experiment:
    """
    Returns an existing MLflow experiment if it exists, otherwise it creates a new one.

    Args:
        experiment_name (str): The name or ID of the experiment.

    Returns:
        mlflow.experiment: The found or created experiment.
    """

    # Look for experiement in existing mlflow experiments
    experiments = mlflow.list_experiments()
    experiment = None
    for experiment in experiments:
        if experiment_name in [experiment.name, experiment.id]:
            return experiment

    # if not found, then create a new one with that name
    return mlflow.create_experiment(name=experiment_name)


class TorchAppMlflowCallback(Callback):
    def __init__(
        self,
        app,
        output_dir: Optional[Path] = None,
        experiment_name: Optional[str] = None,
        log_models: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.app = app

        # tracking_uri will be set if an output directory is given
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            mlflow.set_tracking_uri(f'file://{output_dir.resolve()}')

        # if no experiment_name is given, then it should use the app's project name
        if experiment_name is None:
            experiment_name = app.project_name()

        self.experiment = get_or_create_experiment(experiment_name)

        mlflow.start_run(experiment_id=self.experiment.id)

        mlflow.fastai.autolog(log_models=log_models)

        # checking functions
        self.run = mlflow.active_run()
        console.print(f"Active run_id: {self.run.info.run_id}")

        tracking_uri = mlflow.get_tracking_uri()
        console.print(f"Current tracking URI: {tracking_uri}")

    def after_fit(self):
        mlflow.end_run()

    ################################################
    # The code below isn't used, should it stay here?
    ################################################

    def log(self, param: dict, parameter_metric: bool = False, step: Optional[int] = None):
        """
        Log a dictionary of parameters.

        If parameter metric = True, log as a set of metric with an optional argument 'step'.

        Args:
            param (dict): The dictionary of parameters to be logged.
            parameter_metric (bool, optional): If True, log as a set of metrics. Defaults to False.
            step (Optional[int], optional): an optional argument. Defaults to None.
        """
        if parameter_metric == True:
            mlflow.log_metrics(param, step=step)
        else:
            mlflow.log_params(param)

    def log_artifact(
        self,
        artifact,
        artifact_path: Union[Path, str],
        **kwargs,
    ):
        """
        Input an artifact (pandas dataframe/matplotlib/plotly.figure/dict/str/path) to a saved file

        Args:
            artifact (_type_): artifact to be logged
            artifact_path (Union[Path, str]): path to file to be uploaded.
        """

        if isinstance(artifact, pd.DataFrame):
            csv_file = artifact.to_csv(None, sep='\t')
            mlflow.log_text(csv_file, artifact_path)

        elif isinstance(artifact, dict):
            mlflow.log_dict(artifact, artifact_path)

        elif isinstance(artifact, str):
            mlflow.log_text(artifact, artifact_path)

        elif isinstance(artifact, matplotlib.figure.Figure):
            mlflow.log_figure(artifact, artifact_path)

        #         elif isinstance(artifact, plotly.graph_objs._figure.Figure ):
        #             mlflow.log_figure(artifact, artifact_path)

        #         elif isinstance(artifact, ):
        #             mlflow.log_artifact()

        else:
            pickle.dump(artifact, open(artifact_path, 'wb'))
