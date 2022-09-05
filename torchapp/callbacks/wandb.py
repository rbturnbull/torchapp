import time
import wandb
from fastai.callback.wandb import WandbCallback


class TorchAppWandbCallback(WandbCallback):
    def __init__(self, app, project_name=None, **kwargs):
        self.app = app
        if project_name is None:
            project_name = app.project_name()

        self.run = wandb.init(project=project_name, reinit=True, **kwargs)

        super().__init__()

        if hasattr(app, 'training_kwargs'):
            wandb.config.update(app.training_kwargs)

    def after_epoch(self):
        super().after_epoch()

        # Record the length of time of each epoch
        wandb.log(
            {"time": time.time() - self.recorder.start_epoch},
            step=self._wandb_step,
        )
