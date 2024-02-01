Quick Tour
#######################################################

Below we outline a few examples of using DataDreamer for various use cases to help you get started. It is by no means exhaustive, but should give you a good idea of what
is possible with DataDreamer and how to use it. For more details on the various components of DataDreamer, please see the :doc:`Overview Guide <../overview_guide>`.

Synthetic Data Generation
=========================

- :doc:`Training an "Abstract to Tweet Model" with Fully Synthetic Data <abstract_to_tweet>`
- :doc:`Generating Training Data with Attributed Prompts <attributed_prompts>`
- :doc:`Distilling GPT-4 Capabilities to GPT-3.5 <openai_distillation>`
- :doc:`Augmenting an Existing Dataset <dataset_augmentation>`
- :doc:`Cleaning an Existing Dataset <dataset_cleaning>`
- :doc:`Bootstrapping Synthetic Few-Shot Examples <bootstrapping_machine_translation>`

Instruction-Tuning and Aligning Models
======================================

- :doc:`Instruction-Tuning a LLM <instruction_tuning>`
- :doc:`Aligning a LLM with Human Preferences (RLHF) <aligning>`
- :doc:`Training a Self-Improving LLM with Self-Rewarding (RLAIF) <self_rewarding>`



.. toctree::
    :hidden:
    
    Synthetic Data Generation <https://datadreamer.dev/docs/latest/pages/get_started/quick_tour/index.html#synthetic-data-generation>
    ../motivation_and_design
    abstract_to_tweet
    attributed_prompts
    openai_distillation
    dataset_augmentation
    dataset_cleaning
    bootstrapping_machine_translation
    Instruction-Tuning and Aligning Models <https://datadreamer.dev/docs/latest/pages/get_started/quick_tour/index.html#instruction-tuning-and-aligning-models>
    instruction_tuning
    aligning
    self_rewarding