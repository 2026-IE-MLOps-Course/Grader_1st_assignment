# --------------------------------------------------------
# 1. Ignore internal files but keep the folders tracked
# --------------------------------------------------------
outputs/*
!outputs/.gitkeep

feedback/*
!feedback/.gitkeep

# --------------------------------------------------------
# 2. Specific hidden files to exclude
# --------------------------------------------------------
.pylintrc
.repos copy.text
.scripts.txt

# --------------------------------------------------------
# 3. Standard Python & VS Code clutter (Recommended)
# --------------------------------------------------------
__pycache__/
*.py[cod]
.vscode/
.venv/
venv/
.env
.pytest_cache/