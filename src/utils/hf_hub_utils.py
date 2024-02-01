import json
import re
from io import BytesIO
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable

from .import_utils import ignore_pydantic_warnings

with ignore_pydantic_warnings():
    from huggingface_hub import HfApi, hf_hub_download, login
    from huggingface_hub.utils._errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    from huggingface_hub.utils._headers import LocalTokenNotFoundError
    from huggingface_hub.utils._validators import HFValidationError

from datasets.utils.version import Version

from .. import __version__

if TYPE_CHECKING:  # pragma: no cover
    from ..trainers import Trainer


def _has_file(
    repo_id: str,
    filename: str,
    repo_type: None | str = None,
    revision: None | str | Version = None,
) -> bool:
    try:
        hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision
        )
        return True
    except (
        RepositoryNotFoundError,
        EntryNotFoundError,
        RevisionNotFoundError,
        OSError,
        HFValidationError,
    ):
        return False


def _get_url_to_file(
    repo_id: str,
    filename: str,
    repo_type: None | str = None,
    revision: None | str | Version = None,
) -> None | str:
    if _has_file(
        repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision
    ):
        if repo_type == "dataset":  # pragma: no cover
            return (
                f"https://huggingface.co/datasets/{repo_id}/blob"
                f"/{revision or 'main'}/{filename}"
            )
        else:
            return (
                f"https://huggingface.co/{repo_id}/blob/{revision or 'main'}/{filename}"
            )
    else:
        return None


def get_readme_contents(
    repo_id: str, repo_type: None | str = None, revision: None | str | Version = None
) -> str:
    try:
        local_readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type=repo_type,
            revision=revision,
        )
        with open(local_readme_path, "r") as f:
            return f.read()
    except (
        RepositoryNotFoundError,
        EntryNotFoundError,
        RevisionNotFoundError,
        OSError,
        HFValidationError,
    ):  # pragma: no cover
        return ""


def get_model_card_url(model_name: str) -> None | str:
    return (
        f"https://huggingface.co/{model_name}"
        if _has_file(repo_id=model_name, filename="README.md", repo_type="model")
        else None
    )


def get_license_info(
    repo_id: str, repo_type: None | str = None, revision: None | str | Version = None
) -> None | str:
    license_urls = [
        url
        for url in [
            _get_url_to_file(
                repo_id=repo_id,
                filename="LICENSE.txt",
                repo_type=repo_type,
                revision=revision,
            ),
            _get_url_to_file(
                repo_id=repo_id,
                filename="LICENSE.md",
                repo_type=repo_type,
                revision=revision,
            ),
            _get_url_to_file(
                repo_id=repo_id,
                filename="LICENSE",
                repo_type=repo_type,
                revision=revision,
            ),
        ]
        if url is not None
    ]
    if len(license_urls) > 0:
        return license_urls[0]
    else:
        readme_contents = get_readme_contents(
            repo_id=repo_id, repo_type=repo_type, revision=revision
        )
        search_1_match = re.search(
            re.compile("^license: (.*)$", flags=re.MULTILINE), readme_contents
        )
        search_2_match = re.search(
            re.compile(r"^license:.*$\n\s*-\s*(.*)$", flags=re.MULTILINE),
            readme_contents,
        )
        if search_1_match:
            return search_1_match.group(1)
        elif search_2_match:
            return search_2_match.group(1)
        else:
            return None  # pragma: no cover


def get_citation_info(
    repo_id: str, repo_type: None | str = None, revision: None | str | Version = None
) -> None | list[str]:
    readme_contents = get_readme_contents(
        repo_id=repo_id, repo_type=repo_type, revision=revision
    )
    citations = []
    while match := re.search(r"@(\w+)\s*{\s*", readme_contents):
        readme_contents = readme_contents[match.start() :]
        brace_level = 0
        reached_open = False
        escaping = False
        for idx, c in enumerate(readme_contents):
            if c == "{" and not escaping:
                brace_level += 1
                reached_open = True
            if c == "}" and not escaping:
                brace_level -= 1
            if reached_open and brace_level == 0:
                citations.append(readme_contents[: idx + 1].strip())
                break
            if c == "\\" and not escaping:  # pragma: no cover
                escaping = True
            else:
                escaping = False
        readme_contents = readme_contents[idx + 1 :]
    if len(citations) == 0:  # pragma: no cover
        return None
    else:
        return citations


def hf_hub_login(token: None | str = None) -> HfApi:  # pragma: no cover
    # Login
    api = HfApi()
    if token is not None:
        try:
            login(token=token, add_to_git_credential=False, write_permission=True)
        except ValueError:
            pass
    while True:
        try:
            api.whoami()
            break
        except LocalTokenNotFoundError:
            try:
                login(token=token, add_to_git_credential=False, write_permission=True)
            except ValueError:
                pass
    return api


def prepare_to_publish(
    step_metadata: None | dict[str, Any],
    api: "HfApi",
    repo_id: str,
    repo_type: str,
    branch: None | str = None,
    is_synthetic: bool = True,
) -> tuple[list[str], list[str], list[str], Callable]:  # pragma: no cover
    from ..steps.data_card import DataCardType

    tags = ["datadreamer", f"datadreamer-{__version__}"]
    if is_synthetic:
        tags.append("synthetic")
    if step_metadata is not None:
        dataset_names = list(
            chain.from_iterable(
                [
                    step_metadata["data_card"][step_name].get(
                        DataCardType.DATASET_NAME, []
                    )
                    for step_name in step_metadata["data_card"]
                ]
            )
        )
        model_names = list(
            chain.from_iterable(
                [
                    step_metadata["data_card"][step_name].get(
                        DataCardType.MODEL_NAME, []
                    )
                    for step_name in step_metadata["data_card"]
                ]
            )
        )

        def upload_metadata(trainer: "None | Trainer" = None):
            assert step_metadata is not None
            orig_step_metadata = step_metadata.copy()
            if trainer:
                metadata = trainer._model_card
            else:
                metadata = orig_step_metadata
            api.upload_file(
                path_or_fileobj=BytesIO(bytes(json.dumps(metadata, indent=4), "utf8")),
                path_in_repo="datadreamer.json",
                repo_id=repo_id,
                repo_type=repo_type,
                revision=branch,
                commit_message="Pushed by DataDreamer",
                commit_description="Update datadreamer.json",
            )

    else:
        dataset_names = []
        model_names = []

        def upload_metadata(trainer: "None | Trainer" = None):
            pass

    return tags, dataset_names, model_names, upload_metadata
