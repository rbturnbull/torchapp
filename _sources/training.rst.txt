=======================
Training
=======================

Define the application's ``setup``, ``data``, ``model``, and
``loss_function`` methods before calling ``train``; add ``metrics`` and
override ``monitor`` when model selection should use something other than
validation loss. Start with a short run using ``--max-epochs`` and a dedicated
``--output-dir`` to verify the data and model paths, then increase the training
duration once the logged losses and metrics behave as expected. TorchApp
creates CSV logs and monitored checkpoints in the output directory, so keep
each experiment in a separate directory and retain a full checkpoint whenever
the run may need to be resumed.

Checkpoint modes
================

``train`` supports two mutually exclusive checkpoint modes:

* ``--pretrained-checkpoint PATH`` initializes only the application model
  weights from a Lightning checkpoint. Training starts as a new run with a
  fresh optimizer, scheduler, epoch, and global step.
* ``--resume-checkpoint PATH`` delegates full-state restoration to Lightning,
  including the model, optimizer, scheduler, epoch, global step, callbacks,
  and training-loop state.

Full-state checkpointing writes ``last.ckpt`` in addition to monitored
top-K checkpoints. Use ``last.ckpt`` to resume the latest interrupted run.

Custom pretrained weight transfer
---------------------------------

By default, pretrained loading requires the checkpoint's ``model.*`` state to
match the new model exactly. Applications that need to rename, filter, or
partially load parameters can override ``transfer_pretrained_weights``:

.. code-block:: python

   class MyApp(TorchApp):
       def transfer_pretrained_weights(
           self,
           model,
           pretrained_model_state,
           checkpoint=None,
           **kwargs,
       ):
           remapped_state = {
               name.removeprefix("backbone."): value
               for name, value in pretrained_model_state.items()
               if name.startswith("backbone.")
           }
           model.load_state_dict(remapped_state, strict=False)

``pretrained_model_state`` contains only checkpoint entries below the
``model.`` prefix, with that prefix removed. ``checkpoint`` is the complete
loaded Lightning checkpoint and can provide transfer metadata when needed.
The hook runs after the dummy forward has initialized lazy parameters and
before the trainer is constructed. Overriding it changes only pretrained
initialization; resume mode continues to delegate complete state restoration
to Lightning.
