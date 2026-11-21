# TrustGauge

<p align="center">
  <strong>Evaluating uncertainty and detecting potential hallucinations in Large Language Models.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-TrustGauge-blue" alt="TrustGauge"/>
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/>
</p>

---

## Overview

**TrustGauge** is a toolkit for evaluating uncertainty in Large Language Model (LLM) text generation. It brings together uncertainty estimation (UE) methods that help identify potentially unreliable or hallucinated responses.

By measuring uncertainty at both sequence and claim levels, TrustGauge aims to make LLM evaluation more transparent and support the development of safer, more reliable AI applications.

TrustGauge builds on established research in uncertainty quantification and hallucination detection, incorporating methods and evaluation approaches from the LM-Polygraph research project.

## Features

- Multiple uncertainty estimation methods for LLMs.
- Support for white-box and black-box language models.
- Sequence-level and claim-level uncertainty estimation.
- Compatibility with Hugging Face models.
- Support for OpenAI-compatible API services.
- Benchmarking and evaluation of uncertainty estimation methods.
- Integration with modern LLM inference frameworks.

## Installation

### From GitHub

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/debugcat-hub/trustgauge.git
cd trustgauge

python -m venv env
```

Activate the environment:

**Windows:**
```bash
env\Scripts\activate
```

**Linux/macOS:**
```bash
source env/bin/activate
```

Install the project:

```bash
pip install -e .
```

The existing Python package and module namespace retain the `lm_polygraph` name for compatibility with the underlying implementation.

### Optional dependencies

Some features require additional packages that are not installed by default.

**COMET metric:**

```bash
pip install lm-polygraph[comet]
```

If you need NumPy 2.x compatibility:

```bash
pip install lm-polygraph
pip install unbabel-comet --no-deps
```

## Basic Usage

### 1. Initialize the model

Load a Hugging Face model and tokenizer:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel

model_path = "Qwen/Qwen2.5-0.5B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:0"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = WhiteboxModel(
    base_model,
    tokenizer,
    model_path=model_path
)
```

### 2. Select an uncertainty estimation method

```python
from lm_polygraph.estimators import *

ue_method = MeanTokenEntropy()
```

### 3. Generate uncertainty scores

```python
from lm_polygraph.utils import estimate_uncertainty

input_text = "Who is George Bush?"

ue = estimate_uncertainty(
    model,
    ue_method,
    input_text=input_text
)

print(ue)
```

The output contains the uncertainty score, input text, generated response, and model information.

## Using with LLMs deployed as a service

TrustGauge also supports OpenAI-compatible API services.

```python
from lm_polygraph import BlackboxModel
from lm_polygraph.estimators import Perplexity, MaximumSequenceProbability

model = BlackboxModel.from_openai(
    openai_api_key="YOUR_API_KEY",
    model_path="gpt-4o",
    supports_logprobs=True
)

ue_method = Perplexity()

estimate_uncertainty(
    model,
    ue_method,
    input_text="What has a head and a tail but no body?"
)
```

Certain methods, such as `EigValLaplacian()`, support black-box LLMs that do not provide logits.

## Uncertainty Estimation Methods

TrustGauge incorporates several categories of uncertainty estimation methods:

| Category | Methods |
|---|---|
| Information-based | Perplexity, Mean Token Entropy, Maximum Sequence Probability |
| Meaning Diversity | Semantic Entropy, NumSets, EigenScore |
| Ensembling | Sentence-level and Token-level Ensemble Methods |
| Density-based | Mahalanobis Distance, Robust Density Estimation |
| Reflexive | p(True), Verbalized Uncertainty |

These methods provide different approaches to measuring uncertainty and identifying potentially unreliable model outputs.

## Benchmarking

TrustGauge supports benchmarking uncertainty estimation methods across different datasets and language models.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    model.path=meta-llama/Llama-3.1-8B \
    subsample_eval_dataset=100
```

### Using vLLM

For faster inference using vLLM:

```bash
CUDA_VISIBLE_DEVICES=0 polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    model=vllm \
    model.path=meta-llama/Llama-3.1-8B \
    estimators=default_estimators_vllm \
    stat_calculators=default_calculators_vllm \
    subsample_eval_dataset=100
```

## Examples

The original implementation includes several example notebooks:

- `basic_example.ipynb` – Scoring individual queries.
- `low_level_example.ipynb` – Low-level integration and claim-level uncertainty estimation.
- `low_level_vllm_example.ipynb` – Integration using vLLM.
- `basic_visual_llm_example.ipynb` – Uncertainty estimation for visual language models.

## Research Background

TrustGauge builds on research in uncertainty quantification for Large Language Models.



## License

This project retains the MIT licensing reference from the original implementation. See the [LICENSE](https://github.com/debugcat-hub/trustgauge/blob/main/LICENSE.md) file for details.

---

**TrustGauge — Making LLM reliability measurable.**
