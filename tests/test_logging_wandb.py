from torchapp.examples.iris import IrisApp
import tempfile
import wandb
import os


def test_default_no_wandb():
    app = IrisApp()
    trainer = app.trainer()

    wandb_logger = None
    for logger in trainer._loggers:
        if "WandbLogger" in str(logger):
            wandb_logger = logger
            break

    assert wandb_logger is None


def test_wandb_init():
    app = IrisApp()
    trainer = app.trainer(wandb=True, wandb_offline=True)

    wandb_logger = None
    for logger in trainer._loggers:
        if "WandbLogger" in str(logger):
            wandb_logger = logger
            break


    assert wandb_logger is not None
    assert wandb_logger._offline


def test_wandb_after_epoch():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = IrisApp()
        app.train(wandb=True, wandb_offline=True, wandb_dir=tmpdir, max_epochs=1, output_dir=tmpdir)
        assert isinstance(wandb.summary['epoch_time'], float)
        assert isinstance(wandb.summary['accuracy'], float)
        wandb.finish()  # needs to be called before deleting tmpdir


def test_wandb_kwargs():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = IrisApp()
        trainer = app.trainer(
            wandb=True,
            wandb_offline=True,
            wandb_dir=tmpdir,
            wandb_entity="Entity",
        )
        wandb_logger = None
        for logger in trainer._loggers:
            if "WandbLogger" in str(logger):
                wandb_logger = logger
                break

        assert wandb_logger._wandb_init['project'] == "IrisApp"
        assert wandb_logger._offline == True
        assert os.environ["WANDB_ENTITY"] == "Entity"
        assert str(wandb_logger.save_dir) == str(tmpdir)
