# Update Dependencies once files in dep folder are updated
## Make sure you have the necessary tools installed:

```bash
pip install setuptools wheel
```

## Navigate to the directory containing setup.py in your terminal.
Run the following command to create the wheel distribution:

```bash
python setup.py bdist_wheel
```

## The wheel file will be located in the dist directory.
The file will have a name like llm_bot_dep-0.1.0-py3-none-any.whl, reflecting the package name, version, and other metadata.

## Copy the wheel file to the whl folder for CDK update