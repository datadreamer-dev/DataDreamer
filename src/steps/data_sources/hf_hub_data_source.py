from functools import cached_property

from datasets import DatasetDict, IterableDatasetDict, load_dataset
from datasets.fingerprint import Hasher
from datasets.splits import Split
from datasets.utils.version import Version

from ...utils.hf_hub_utils import _has_file, get_citation_info, get_license_info
from ..data_card import DataCardType
from ..step_operations import _INTERNAL_STEP_OPERATION_KEY
from .data_source import DataSource


class HFHubDataSource(DataSource):
    def __init__(
        self,
        name: str,
        path: str,
        config_name: None | str = None,
        split: None | str | Split = None,
        revision: None | str | Version = None,
        trust_remote_code: bool = False,
        streaming: bool = False,
        progress_interval: None | int = None,
        force: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
        **config_kwargs,
    ):
        """Loads a dataset from the Hugging Face Hub. See
        :py:func:`datasets.load_dataset` for more details.

        Args:
            name: The name of the step.
            path: The path to the dataset on the Hugging Face Hub.
            config_name: The name of the dataset configuration to load.
            split: The split to load.
            revision: The version (commit hash) of the dataset to load.
            trust_remote_code: Whether to trust the remote code.
            streaming: Whether to load the dataset in streaming mode.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            verbose: Whether or not to print verbose logs.
            log_level: The logging level to use (:py:data:`~logging.DEBUG`, :py:data:`~logging.INFO`, etc.).
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.
        """
        self.path = path
        self.config_name = config_name
        self.split = split
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.streaming = streaming
        self.config_kwargs = config_kwargs
        super().__init__(
            name,
            data=None,  # type: ignore[arg-type]
            progress_interval=progress_interval,
            force=force,
            verbose=verbose,
            log_level=log_level,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
            background=background,
        )

    def setup(self):
        pass

    def run(self):
        from ... import DataDreamer

        DataDreamer._enable_hf_datasets_logging()
        result = load_dataset(
            path=self.path,
            name=self.config_name,
            split=self.split,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            streaming=self.streaming,
            **self.config_kwargs,
        )
        DataDreamer._disable_hf_datasets_logging()
        if isinstance(result, (DatasetDict, IterableDatasetDict)):
            raise ValueError(
                "Choose a split with the `split=...` parameter,"
                " multiple splits are not supported."
            )
        if _has_file(repo_id=self.path, filename="README.md", repo_type="dataset"):
            self.register_data_card(DataCardType.DATASET_NAME, self.path)
            self.register_data_card(
                DataCardType.DATASET_CARD,
                f"https://huggingface.co/datasets/{self.path}",
            )
        if hasattr(result, "homepage") and result.homepage:  # pragma: no cover
            self.register_data_card(DataCardType.URL, result.homepage)
        if license_info := get_license_info(
            self.path, repo_type="dataset", revision=self.revision
        ):
            self.register_data_card(DataCardType.LICENSE, license_info)
        elif hasattr(result, "license") and result.license:  # pragma: no cover
            self.register_data_card(DataCardType.LICENSE, result.license)
        if hasattr(result, "citation") and result.citation:  # pragma: no cover
            self.register_data_card(DataCardType.CITATION, result.citation)
        elif citation_info := get_citation_info(
            self.path, repo_type="dataset", revision=self.revision
        ):
            for citation in citation_info:
                self.register_data_card(DataCardType.CITATION, citation)
        return result

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash(
            [
                super().fingerprint,
                self.path,
                self.config_name,
                self.split,
                self.revision,
                self.streaming,
                self.config_kwargs,
            ]
        )


setattr(HFHubDataSource, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["HFHubDataSource"]
