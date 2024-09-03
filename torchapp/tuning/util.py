import lightning as L

def best_model_score(trainer:L.Trainer) -> float:
    assert getattr(trainer, "checkpoint_callback", None) is not None, "Trainer must have a checkpoint callback"
    return trainer.checkpoint_callback.best_model_score.item()