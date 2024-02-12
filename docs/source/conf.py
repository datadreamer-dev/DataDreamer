import importlib
import inspect
import itertools
import os
import platform
import sys
from datetime import datetime
from functools import cached_property, partial

from sphinx.util.inspect import signature, stringify_signature

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from source import docstrings

# Add path to source files
sys.path.insert(0, "../")


# Read project information
def read_toml_string(toml_lines, key):
    line = [line.strip() for line in toml_lines if line.startswith(key)][0]
    return line[line.find('"') + 1 : line.rfind('"')]


with open("../../pyproject.toml") as pyproject_fp:
    pyproject_lines = [line.replace("# ", "").strip() for line in pyproject_fp]
    package_name = read_toml_string(pyproject_lines, "name").lower()
    __version__ = read_toml_string(pyproject_lines, "version")
    documentation_url = read_toml_string(pyproject_lines, "documentation")
    repository_url = read_toml_string(pyproject_lines, "repository")
    docs_author = read_toml_string(pyproject_lines, "docs_author")
    docs_author_website = read_toml_string(pyproject_lines, "docs_author_website")

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
author = docs_author
author_website = docs_author_website
copyright = f"{datetime.now().year}, {author}"
release = __version__
github_username = repository_url.split("github.com/")[1].split("/")[0]
github_repository = project
source_repository = f"https://github.com/{github_username}/{github_repository}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    # 'sphinx_toolbox.more_autodoc.typehints',
    "sphinx_autodoc_typehints",
    "sphinx.ext.linkcode",
    "sphinx.ext.todo",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "sphinx_toolbox.changeset",
    "sphinx_toolbox.code",
    "sphinx_toolbox.collapse",
    "sphinx_toolbox.decorators",
    "sphinx_toolbox.github",
    "sphinx_toolbox.wikipedia",
    "sphinx_click",
    "sphinx_sitemap",
    "sphinx_design",
    "sphinx_reredirects",
]
root_doc = "index"
redirects = {
     "pages/get_started/installation": "/#installation",
}
exclude_patterns = []
coverage_show_missing_items = True
templates_path = ["_templates"]
modindex_common_prefix = [f"{package_name}."]
add_module_names = True
todo_include_todos = True
sitemap_url_scheme = "{link}"
python_version = ".".join(platform.python_version_tuple()[0:2])
intersphinx_mapping = {
    "python": ("https://docs.python.org/" + python_version, None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
    "peft": ("https://huggingface.co/docs/peft/main/en/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "dill": ("https://dill.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = project
html_baseurl = documentation_url + "latest/"
html_logo = "_static/images/logo.svg"
html_favicon = "_static/images/favicon.svg"
html_show_sphinx = False
html_static_path = ["_static"]
html_js_files = [
    "https://buttons.github.io/buttons.js",
    "js/custom.js"
]
html_css_files = [
    "css/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
html_theme_options = {
    "source_repository": source_repository,
    "source_branch": "main",
    "source_directory": "docs/source",
    "footer_icons": [
        {
            "name": "Personal Website",
            "url": author_website,
            "html": "👨🏽‍💻",
            "class": "footer-emojis",
        },
        {
            "name": "GitHub",
            "url": source_repository,
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
    ],
    "light_css_variables": {
        "color-api-pre-name": "#8C1AF5",
        "color-api-name": "#C07373",
        "color-link-underline": "#d4ccd4",
    },
}
pygments_style = "nord"
pygments_dark_style = "nord"

# -- Linkcode configuration --------------------------------------------------
code_url = f"{source_repository}/blob/{__version__}"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        obj_to_inspect = obj
        obj_to_inspect = (
            obj_to_inspect.func
            if isinstance(obj_to_inspect, cached_property)
            else obj_to_inspect
        )
        obj_to_inspect = (
            obj_to_inspect.fget
            if isinstance(obj_to_inspect, property)
            else obj_to_inspect
        )
        file = inspect.getsourcefile(obj_to_inspect)
        lines = inspect.getsourcelines(obj_to_inspect)
    except Exception:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))
    if not file.startswith(f"{package_name}/"):
        return None
    file = file.replace(f"{package_name}/", "src/")
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{code_url}/{file}#L{start}-L{end}"


# -- AutoDoc configuration --------------------------------------------------
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True
autodoc_default_options = {"inherited-members": True}
typehints_use_signature = False
typehints_fully_qualified = False
autodoc_typehints = "signature"
typehints_defaults = "comma"
always_document_param_types = False


def get_class_that_defined_method(meth):
    try:
        vals = vars(sys.modules[meth.__module__])
        for attr in meth.__qualname__.split(".")[:-1]:
            vals = vals[attr]
        return vals
    except Exception:
        pass
    return meth.__class__


def autodoc_skip_member(seen, app, what, name, obj, skip, options):
    obj = obj.func if isinstance(obj, cached_property) else obj
    obj = obj.fget if isinstance(obj, property) else obj
    qualname = obj.__qualname__ if hasattr(obj, "__qualname__") else name
    datadreamer_excludes = [
        "ctx",
        "is_background_process",
        "is_registered_thread",
        "is_running_in_memory",
    ]
    step_excludes = ["setup", "fingerprint", "with_traceback", "help"]
    cachable_excludes = [
        "cache_and_lock",
        "get_logger",
        "reset_adaptive_batch_sizing",
        "display_name",
        "display_icon",
        "license",
        "base_model_card",
        "model_card",
        "citation",
        "version",
    ]
    llm_excludes = [
        "final_count_tokens",
        "retry_wrapper",
        "tokenizer_model_name",
        "config",
    ]
    trainer_excludes = ["resumable", "auto_cls", "compute_fingerprint"]
    qualname_excludes = [
        "OutputDataset.save_to_disk",
        "OutputDatasetMixin.head",
        "OutputDatasetMixin.info",
        "BaseException.args",
        "BaseException.add_note",
        "TrainSentenceTransformer.train",
        "TrainHFRewardModel.train",
        "LazyRows.value",
        "LazyRowBatches.value",
        "ParallelEmbedder.dims",
        "ParallelEmbedder.model_max_length",
        "ParallelRetriever.index",
        "TrainOpenAIFineTune.client",
        "TrainOpenAIFineTune.tokenizer",
        "TrainOpenAIFineTune.get_max_context_length",
        "TrainOpenAIFineTune.count_tokens",
    ]
    qualname_includes = [
        "OutputDatasetMixin.__getitem__",
        "Trainer.model_card",
        "Embedder.count_tokens"
    ]
    exclusions = set(
        itertools.chain.from_iterable(
            [
                datadreamer_excludes,
                step_excludes,
                cachable_excludes,
                llm_excludes,
                trainer_excludes,
            ]
        )
    )
    excluded_via_private = name.startswith("_")
    excluded_via_exclusions = name in exclusions
    is_step = hasattr(get_class_that_defined_method(obj), "setup")
    excluded_via_step_run = name == "run" and is_step
    excluded_via_qualname = qualname in qualname_excludes
    try:
        obj_cls = get_class_that_defined_method(obj)
        if isinstance(obj, (property, cached_property)):
            excluded_via_inherited = any(
                hasattr(parent_cls, name) for parent_cls in obj_cls.__bases__
            )
        else:
            excluded_via_inherited = any(
                hasattr(parent_cls, name)
                and stringify_signature(signature(obj, bound_method=True))
                == stringify_signature(
                    signature(getattr(parent_cls, name), bound_method=True)
                )
                for parent_cls in obj_cls.__bases__
            )
    except Exception:
        excluded_via_inherited = False
    if str(obj) in seen:
        excluded_via_inherited = True
    excluded_via_heuristics = (
        excluded_via_private
        or excluded_via_exclusions
        or excluded_via_step_run
        or excluded_via_qualname
        or excluded_via_inherited
    )
    always_include = ["model", "model_path", "tokenizer", "client", "index", "output"]
    always_include_verbs = ["run", "train", "publish", "export", "output"]
    included_via_qualname = qualname in qualname_includes
    included_via_always = name in always_include
    included_via_always_verbs = not is_step and any(
        name == v or name.startswith(f"{v}_") for v in always_include_verbs
    )
    included = included_via_qualname or (
        not excluded_via_qualname and (included_via_always or included_via_always_verbs)
    )
    if excluded_via_heuristics and included and (not included_via_qualname or not excluded_via_inherited or qualname in ["Embedder.count_tokens"]):
        excluded = False
    else:
        excluded = excluded_via_heuristics
    seen.add(str(obj))
    return excluded


def autodoc_process_docstring(app, what, name, obj, options, lines):
    objname = name.split(".")[-1]
    if (
        name.startswith("datadreamer.steps.")
        and hasattr(obj, "help")
        and not name.endswith(".Step")
        and not name.endswith(".SuperStep")
        and not name.endswith("DataSource")
    ):
        docstring = docstrings.STEP_HELP.replace("CLS_NAME", objname)
        docstrings.append(lines, docstring)
        docstrings.append_new_lines(lines, 1)
        docstrings.append(lines, obj.help, indent_level=1)
    elif objname in ["model", "tokenizer", "client", "assistant_id", "index", "dims", "model_max_length"] and lines == []:
        docstrings.clear(lines)
        docstrings.append(lines, ({
            "model": "The model instance being used.",
            "tokenizer": "The tokenizer instance being used.",
            "client": "The API client instance being used.",
            "assistant_id": "The ID of the assistant.",
            "index": "The index instance being used.",
            "dims": "The dimensions of the embeddings.",
            "model_max_length": "The maximum length of the model.",
        })[objname])
    elif objname == "run":
        if "queries" in signature(obj).parameters:
            lines.clear()
            lines.extend(docstrings.RETRIEVER_RUN)
        if "texts" in signature(obj).parameters:
            lines.clear()
            if "instruction" in signature(obj).parameters:
                lines.extend(docstrings.TASK_MODEL_RUN_WITH_INSTRUCTION)
            else:
                lines.extend(docstrings.TASK_MODEL_RUN)


def setup(app):
    app.connect("autodoc-skip-member", partial(autodoc_skip_member, set()))
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
