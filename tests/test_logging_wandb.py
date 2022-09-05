from torchapp.callbacks.wandb import TorchAppWandbCallback, wandb
from torchapp.examples.iris import IrisApp
import tempfile


def test_default_no_wandb():
    app = IrisApp()
    callbacks = app.callbacks()

    callback = None
    for c in callbacks:
        if isinstance(c, TorchAppWandbCallback):
            callback = c
            break
    assert callback is None
    assert wandb.run is None


def test_wandb_init():
    app = IrisApp()
    callbacks = app.callbacks(wandb=True, wandb_mode="disabled")

    callback = None
    for c in callbacks:
        if isinstance(c, TorchAppWandbCallback):
            callback = c
            break
    assert callback is not None
    assert wandb.run is not None
    assert callback.run is wandb.run


def test_wandb_after_epoch():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = IrisApp()
        app.train(wandb=True, wandb_mode="offline", wandb_dir=tmpdir, epochs=1)
        assert isinstance(wandb.summary['time'], float)
        wandb.finish()  # needs to be called before deleting tmpdir


def test_wandb_kwargs():
    with tempfile.TemporaryDirectory() as tmpdir:
        IrisApp().callbacks(
            wandb=True,
            wandb_mode="offline",
            wandb_dir=tmpdir,
            tag=["Tag1", "Tag2"],
            run_name="Run",
            wandb_group="Group",
            notes="Notes",
            wandb_entity="Entity",
            wandb_job_type="JobType",
        )
        assert wandb.run.tags == ("Tag1", "Tag2")
        assert wandb.run.group == "Group"
        assert wandb.run.name == "Run"
        assert wandb.run.entity == "Entity"
        assert wandb.run.notes == "Notes"
        assert wandb.run.job_type == "JobType"
        wandb.finish()
