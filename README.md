## Development

### Environment Setup

This project uses conda for environment management. To set up the development environment:

```bash
conda env create -f environment.yml
conda activate em_recharge
```

### Code Formatting

This project uses [pre-commit](https://pre-commit.com/) hooks to automatically format code and notebooks with [ruff](https://docs.astral.sh/ruff/).

#### Installing pre-commit

Pre-commit is included in the environment.yml, but if you need to install it separately:

```bash
pip install pre-commit
```

#### Setting up pre-commit hooks

After installing pre-commit, install the git hooks:

```bash
pre-commit install
```

This will configure git to automatically run the formatting checks before each commit.

#### Running pre-commit manually

To run pre-commit on all files manually (useful for first-time setup or checking all files):

```bash
pre-commit run --all-files
```

To run on specific files:

```bash
pre-commit run --files path/to/file.py
```

#### What gets formatted

The pre-commit hooks will automatically:

- Format Python files with ruff (line length: 100 characters)
- Format Jupyter notebooks with ruff via nbqa
- Remove trailing whitespace
- Fix end-of-file formatting
- Check YAML syntax
- Check for large files and merge conflicts

#### Formatting configuration

Formatting rules are defined in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).
