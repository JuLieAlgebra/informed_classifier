# informed_classification
Demonstration of ideas from scientific machine learning and informed ML

## Description

Tokamaks are a donut-shaped nuclear fusion reactor that control plasma with magnetic fields. This controlled state of plasma is subject to ("disruptions")[], which is a wide class of events that result in the plasma becoming uncontrolled and, typically, slamming into the walls of the fusion reactor, causing large amount of damage during high-energy (shots)[]. 

If physicists can detect when the plasma is in a disrupted state, and in the best case, what kind of disrupted state, damage migitation systems can kick in and vastly reduce the amount of damage to the reactor (source)[]. Real-time disruption prediction is dominated by handcrafted, hand-tuned thresholds and Gaussian-Process based methods, with machine learning methods like Random Decision Trees gaining traction (source, source)[].

Many of these disruption events have reliable physics models, but many have incomplete descriptions or are too computationally expensive for real-time deployment. Nominal plasma states are, tautologically, well modeled by a set of equilibrium descriptions like MHD (source)[] are typically modeled as a GP (source)[]. 

This work was inspired by this problem in plasma physics, which is becoming an increasinly important problem to bringing net-energy fusion devices to market as more powerful fusion reactors experience more damage from these events. Directly modeling the boundary between disrupted and nominal states of plasma represents the majority of effort on this subject (source)[], but where does the boundary lie for using marginal models for a discriminative task? When is it best to use modeling parameters for discrimination vs incorporate existing knowledge into modeling the marginal distributions? What if you perfectly knew the distributions of two classes, but nothing about the third? We often have problems where we have much structured knowledge (simulations, mathematical models, etc) on class-specific dynamics, but do not for the joint distribution.

This work is a simplification of this problem for a controlled experiment setting.
Assume that we have a dynamical system (for example, plasma in a tokamak) that can be in one of two hidden states and you wanted to estimate which state the system was in. Trajectory data from this process looks like this:

[image]

There are a wide array of reasonable approaches to take to discriminate between the two -- neural networks, ...
But what if you knew the underlying stochastic process and class distribution for each of the two states? 

### Disrupted

<img src="docs/media/fitted_disrupted_model_progress.gif" width="600" height="500" />

### Nominal

<img src="docs/media/fitted_nominal_model_progress.gif" width="600" height="500" />

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