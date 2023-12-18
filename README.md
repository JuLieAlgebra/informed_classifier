# informed_classification
Demonstration of ideas from scientific machine learning and informed ML

## Setup
Python >=3.9 is required for this repo, it was developed with Python 3.11.5.\

Install Poetry:
```sh
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.3.2 python -
```
Install informed_classification:
```
poetry install
```

Run tests:
```
poetry run pytest
```

## Running
First generate the data:
`poetry run python scripts/gen_data.py --config config_filename_in_config_dir`

Then run the script to evaluate the models based on the generated data:
`poetry run python scripts/evaluate_{MODEL_CLASS}.py --config config_filename_in_config_dir` 