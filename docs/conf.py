import sys
import os

# Add the parent directory to the path so that we can import flood_tool
sys.path.insert(0, os.path.abspath(os.path.join(os.curdir, "..")))

project = "Flood Tool"
author = "Thames"
version = "0.1.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build"]
autoclass_content = "both"
autodoc_mock_imports = ["sklearn","imblearn","re"]

html_theme = 'bizstyle'
