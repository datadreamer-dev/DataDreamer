import importlib
import inspect
import os
import platform
import sys
from datetime import datetime

# Add path to source files
sys.path.insert(0, '../')

# Read project information
def read_toml_string(toml_lines, key):
    line = [line.strip() for line in toml_lines if line.startswith(key)][0]
    return line[line.find('"')+1:line.rfind('"')]

with open("../../pyproject.toml") as pyproject_fp:
    pyproject_lines = [line.replace('# ', '').strip() for line in pyproject_fp]
    package_name = read_toml_string(pyproject_lines, "name")
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
copyright = f'{datetime.now().year}, {author}'
release = __version__
github_username = repository_url.split('github.com/')[1].split('/')[0]
github_repository = project
source_repository = f'https://github.com/{github_username}/{github_repository}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_toolbox.more_autodoc.typehints',
    'sphinx_autodoc_typehints',
    'sphinx.ext.linkcode',
    'sphinx.ext.todo',
    'sphinx_inline_tabs',
    'sphinx_copybutton',
    'sphinx_toolbox.changeset',
    'sphinx_toolbox.code',
    'sphinx_toolbox.collapse',
    'sphinx_toolbox.decorators',
    'sphinx_toolbox.github',
    'sphinx_toolbox.wikipedia',
    'sphinx_click',
    'sphinx_sitemap',
]
root_doc = 'index'
exclude_patterns = []
coverage_show_missing_items = True
templates_path = ['_templates']
modindex_common_prefix = [f'{package_name}.']
add_module_names = True
todo_include_todos = True
sitemap_url_scheme = "{link}"
python_version = '.'.join(platform.python_version_tuple()[0:2])
intersphinx_mapping = {'python': ('https://docs.python.org/'+python_version, None)}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = project
html_baseurl = documentation_url+"latest/"
# html_logo = 'https://www.google.com/images/branding/googlelogo/2x/googlelogo_dark_color_92x30dp.png'
# html_favicon = "https://github.githubassets.com/favicons/favicon.svg"
html_show_sphinx = False
html_static_path = ['_static']
html_js_files = [
    "https://buttons.github.io/buttons.js",
]
html_css_files = [
    'css/custom.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css',
]
html_theme_options = {
    "source_repository": source_repository,
    "source_branch": "main",
    "source_directory": "docs/source",
    'footer_icons': [
        {
            'name': 'Personal Website',
            'url': author_website,
            'html': '👨🏽‍💻',
            'class': 'footer-emojis',
        },
        {
            'name': 'GitHub',
            'url': source_repository,
            'html': '',
            'class': 'fa-brands fa-solid fa-github fa-2x',
        },
    ],
}

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
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))
    if not file.startswith(f"{package_name}/"):
        return None
    file = file.replace(f"{package_name}/", "src/")
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{code_url}/{file}#L{start}-L{end}"