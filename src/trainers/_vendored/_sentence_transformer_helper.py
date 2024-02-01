# type: ignore
# ruff: noqa

import os
import shutil
import stat
import tempfile
from typing import Optional, List

from ... import DataDreamer
from ...utils.import_utils import ignore_pydantic_warnings

with ignore_pydantic_warnings():
    from huggingface_hub import HfApi, HfFolder, Repository


# TODO (fix later if SentenceTransformer updates):
# Due to: https://github.com/UKPLab/sentence-transformers/issues/1925
# This function is pulled in from:
#    https://github.com/UKPLab/sentence-transformers/blob/
#    c006921e9e9977bc107b05676266b581091688a2/sentence_transformers/
#    SentenceTransformer.py#L386


def save_to_hub(
    self,
    repo_id: str,
    organization: Optional[str] = None,
    token: Optional[str] = None,
    private: Optional[bool] = None,
    commit_message: str = "Add new SentenceTransformer model.",
    local_model_path: Optional[str] = None,
    exist_ok: bool = False,
    replace_model_card: bool = False,
    train_datasets: Optional[List[str]] = None,
):  # pragma: no cover
    """
    Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

    :param repo_id: Repository name for your model in the Hub, including the user or organization.
    :param token: An authentication token (See https://huggingface.co/settings/token)
    :param private: Set to true, for hosting a private model
    :param commit_message: Message to commit while pushing.
    :param local_model_path: Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
    :param exist_ok: If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
    :param replace_model_card: If true, replace an existing model card in the hub with the automatically created model card
    :param train_datasets: Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
    :param organization: Deprecated. Organization in which you want to push your model or tokenizer (you must be a member of this organization).

    :return: The url of the commit of your model in the repository on the Hugging Face Hub.
    """
    if organization:
        if "/" not in repo_id:
            logger.warning(
                f'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="{organization}/{repo_id}"` instead.'
            )
            repo_id = f"{organization}/{repo_id}"
        elif repo_id.split("/")[0] != organization:
            raise ValueError(
                "Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id`."
            )
        else:
            logger.warning(
                f'Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id="{repo_id}"` instead.'
            )

    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=repo_id, private=private, repo_type=None, exist_ok=exist_ok
    )
    if local_model_path:
        folder_url = api.upload_folder(
            repo_id=repo_id, folder_path=local_model_path, commit_message=commit_message
        )
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_model_card = replace_model_card or not os.path.exists(
                os.path.join(tmp_dir, "README.md")
            )
            # DataDreamer: We add a custom README, disabling create_model_card
            self.save(
                tmp_dir,
                model_name=repo_url.repo_id,
                create_model_card=False,
                train_datasets=train_datasets,
            )
            folder_url = api.upload_folder(
                repo_id=repo_id, folder_path=tmp_dir, commit_message=commit_message
            )

    refs = api.list_repo_refs(repo_id=repo_id)
    for branch in refs.branches:
        if branch.name == "main":
            return f"https://huggingface.co/{repo_id}/commit/{branch.target_commit}"
    # This isn't expected to ever be reached.
    return folder_url
