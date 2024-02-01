Caching and Saved Outputs
#######################################################

DataDreamer aggressively caches and saves its work at multiple levels to avoid re-computing when possible to be as time- and cost-efficient as possible.

- **Step Outputs**: DataDreamer caches the results of each step run within a session to the output folder. If a session is interrupted and re-run, DataDreamer will automatically load the results of previously completed steps from disk and resume where it left off.
- **Model Generations and Outputs**: DataDreamer caches the results computed by a :py:class:`~datadreamer.llms.LLM`, :py:class:`~datadreamer.embedders.Embedder` model, etc.
- **Training Checkpoints**: DataDreamer will automatically save and resume from checkpoints when training a model with :py:class:`~datadreamer.trainers.Trainer`.

Output Folder File Structure
===============================

:py:class:`~datadreamer.DataDreamer` sessions write to an output folder where all outputs and caches are saved. Below is a brief description of the output folder structure.

- **Step Folders**: Each :py:class:`~datadreamer.steps.Step` will produce a named folder within the output folder. The name of the folder is the name of the step, and the folder contains the output dataset of the step within a ``_dataset`` folder. ``step.json`` contains metadata about the step. If a step is run within another step, its folder will be nested under the parent step's folder.
- **Trainer Folders**: Each :py:class:`~datadreamer.trainers.Trainer` will produce a named folder within the output folder. The name of the folder is the name of the trainer, and the folder contains saved checkpoints during training to a ``_checkpoints`` folder and the final trained model to a ``_model`` folder. Various JSON files inside the ``_model`` folder like ``training_args.json`` contain metadata about the training configuration.
- **Cache Folder**: The ``.cache`` folder in the output folder holds the SQLite databases that are used to cache the generations and outputs produced by models like  :py:class:`~datadreamer.llms.LLM` or :py:class:`~datadreamer.embedders.Embedder`.
- **Backups Folder**: The ``_backups`` folder in the output folder holds backups of step or trainer folders that have since been invalidated by a newer configuration of that step or trainer. They are kept in case a user reverts to a previous configuration of the step or trainer.