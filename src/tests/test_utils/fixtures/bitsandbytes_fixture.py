import warnings

import pytest

imported_banb = False


@pytest.fixture(autouse=True)
def reset_bandb_import():
    """We need to import banb as it throws a warning on first import"""
    global imported_banb

    # Code that will run before your test
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The installed version of bitsandbytes was compiled without GPU.*",
            module="bitsandbytes.cextension",
        )
        if not imported_banb:
            print(
                "\nDataDreamer test suite is importing bitsandbyes,"
                " ignore any warnings below this...\n"
            )
        import bitsandbytes  # noqa: F401

        if not imported_banb:
            print("\nDataDreamer test suite is done importing bitsandbyes.\n")

        imported_banb = True

    yield
    # Code that will run after your test
