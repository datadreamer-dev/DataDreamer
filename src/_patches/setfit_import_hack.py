# SetFit is out-of-date with huggingface_hub and throws an error when trying to import
# from it
# like this: ImportError: cannot import name 'DatasetFilter' from 'huggingface_hub'

# To fix this, we need to monkey patch huggingface_hub to prevent the import error

from ..utils.import_utils import ignore_pydantic_warnings


def apply_setfit_import_hack():
    with ignore_pydantic_warnings():
        import huggingface_hub

        huggingface_hub.DatasetFilter = None
