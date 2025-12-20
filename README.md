# Task Authoring Framework

A framework for creating and evaluating scientific coding tasks. Domain experts can author multi-step coding problems with automatic test generation and evaluation.

## Overview

This framework enables scientists and domain experts to create structured coding challenges that:
- Decompose complex problems into sequential sub-steps
- Include gold-standard reference implementations
- Auto-generate test targets from gold solutions
- Evaluate LLM-generated code against expected outputs

## Quick Start

### 1. Installation

```bash
git clone https://github.com/mercor-code-envs/scicode-extended-template.git
cd scicode-extended-template

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Create a New Task

```bash
# Create task from template
task new my_task_name

# Edit the generated files:
# - tasks/my_task_name/problem.yaml    (metadata)
# - tasks/my_task_name/steps/*.py      (sub-problems + gold solutions)
# - tasks/my_task_name/background/*.md (optional context)
```

### 3. Validate and Compile

```bash
# Validate task structure
task validate tasks/my_task_name/

# Compile to evaluation format (generates eval/data/problems.jsonl and test_data.h5)
task compile tasks/my_task_name/
```

> **Note:** Compiled data files (`problems.jsonl`, `test_data.h5`) are not committed to the repo.
> After cloning, run `task compile-all` to generate them from the task definitions.

### 4. Set Up API Keys

```bash
# Google (Gemini)
export GOOGLE_API_KEY=your-google-api-key

# OpenAI
export OPENAI_API_KEY=your-openai-api-key

# Anthropic
export ANTHROPIC_API_KEY=your-anthropic-api-key

# Together AI
export TOGETHER_API_KEY=your-together-api-key
```

### 5. Evaluate with an LLM

```bash
cd eval/inspect_ai

# With Google Gemini
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=my_task_name \
    --model google/gemini-3-pro-preview

# With OpenAI
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=my_task_name \
    --model openai/gpt-5.2

# With Anthropic
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=my_task_name \
    --model anthropic/claude-sonnet-4-5

# With Together AI
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=my_task_name \
    --model together/meta-llama/Llama-3-70b-chat-hf
```

## Project Structure

```
├── tasks/                      # Task authoring directory
│   ├── _template/              # Template for new tasks
│   └── your_task/              # Your custom tasks
│       ├── problem.yaml        # Task metadata
│       ├── steps/              # Sub-problem definitions
│       │   ├── 01_step.py      # Gold solution + tests
│       │   └── 02_step.py
│       └── background/         # Optional context
│
├── eval/
│   ├── data/                   # Compiled task data
│   │   ├── problems.jsonl      # Task definitions
│   │   └── test_data.h5        # Test targets
│   └── inspect_ai/             # Evaluation harness
│       └── scicode.py          # inspect_ai task definition
│
└── src/scicode/
    ├── authoring/              # Task authoring tools
    │   ├── cli.py              # task CLI
    │   ├── compiler.py         # YAML/Python → JSON/H5
    │   └── validator.py        # Task validation
    └── tools/
        └── inspector.py        # inspect-task CLI
```

## Task Authoring Guide

### Task Structure

Each task consists of:
1. **Main problem** - Overall description and metadata
2. **Sub-steps** - Sequential functions that build on each other
3. **Gold solutions** - Reference implementations for each step
4. **Test cases** - Inputs that validate correctness
5. **Background** (optional) - Scientific context for each step

### Creating Steps

Each step file (`steps/01_name.py`) contains:

```python
"""
Step description shown to the LLM.
Explain what function to implement.
"""
import numpy as np

def my_function(x, y):
    '''Docstring with parameters and return value.'''
    return result

def _gold_my_function(x, y):
    '''Reference implementation - generates test targets.'''
    return x + y  # Actual implementation

def test_cases():
    return [
        {
            "setup": "x = 5\ny = 10",
            "call": "my_function(x, y)",
            "gold_call": "_gold_my_function(x, y)",
        },
    ]
```

### CLI Commands

```bash
task new <name>           # Create from template
task validate <dir>       # Check task structure
task compile <dir>        # Generate JSON + H5
task compile-all          # Compile all tasks
task list                 # List all tasks
task test <dir>           # Compile + run eval

inspect-task              # List compiled tasks
inspect-task <task_id>    # Inspect task structure
```

## Evaluation

### Using inspect_ai

```bash
cd eval/inspect_ai

# Evaluate on your custom tasks
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=your_task_id \
    --model openai/gpt-4o \
    --temperature 0

# Evaluate with scientific background
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T with_background=True \
    --model anthropic/claude-sonnet-4-5
```

### Evaluation Options

| Option | Description |
|--------|-------------|
| `local_jsonl` | Path to compiled problems.jsonl |
| `problems` | Comma-separated task IDs to run |
| `with_background` | Include scientific background in prompts |
| `mode` | `normal`, `gold` (test with solutions), `dummy` |
| `output_dir` | Directory for generated code |

## Collaboration

### Naming Convention

Use descriptive names to avoid merge conflicts:
- `lennard_jones_potential`
- `protein_folding_energy`
- `molecular_dynamics_sim`

### Workflow

```bash
git checkout -b add-task/my_new_task
task new my_new_task
# ... develop task ...
task validate tasks/my_new_task/
git add tasks/my_new_task/
git commit -m "Add my_new_task"
git push origin add-task/my_new_task
# Open PR for review
```

## Documentation

- [Task Authoring Guide](tasks/README.md) - Detailed task creation documentation
- [Evaluation Guide](eval/inspect_ai/README.md) - Running evaluations
