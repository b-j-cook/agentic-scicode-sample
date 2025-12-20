# Task Authoring Directory

This directory contains task definitions for evaluation.

## Quick Start

### Creating a New Task

```bash
# 1. Create a new task from the template
task new my_task_name

# 2. Edit the generated files in tasks/my_task_name/
#    - problem.yaml: Task metadata and structure
#    - steps/*.py: Individual sub-problems with gold solutions
#    - background/*.md: Optional background

# 3. Compile and validate your task
task compile tasks/my_task_name/

# 4. Test with the evaluation harness
task test tasks/my_task_name/
```

## Directory Structure

```
tasks/
├── _template/              # Template for new tasks (DO NOT EDIT)
│   ├── problem.yaml        # Task metadata template
│   ├── steps/              # Step file templates
│   │   └── 01_step_name.py
│   └── background/         # Background templates (optional)
│       └── 01_step_name.md
│
├── your_task_name/         # Your actual tasks
│   ├── problem.yaml        # Required
│   ├── steps/              # Required
│   │   ├── 01_first_step.py
│   │   ├── 02_second_step.py
│   │   └── ...
│   └── background/         # Optional - for with_background=True evaluation
│       ├── 01_first_step.md
│       ├── 02_second_step.md
│       └── ...
│
└── README.md               # This file
```

**Note:** The "general tests" are automatically derived from the last subproblem's 
test cases. The final step implicitly validates all previous steps since they 
build on each other.

**Background files** are optional. If provided, they populate the `step_background` 
field which is shown to models when evaluated with `with_background=True`.

## Naming Convention

**IMPORTANT**: Use descriptive names, not numbers, to avoid merge conflicts!

Format: `{descriptive_name}` or `{category}_{descriptive_name}`

Examples:
- `lennard_jones_potential`
- `protein_folding_energy`
- `reaction_kinetics_solver`
- `eigenvalue_decomposition`
- `crystal_structure_analysis`

## Working with Multiple Contributors

### Avoiding Conflicts

1. **Use descriptive task IDs** - Never use numeric IDs like "77" or "78"
2. **Use unique names** - Descriptive names help organize and prevent collisions
3. **Claim tasks before starting** - Use issues/PRs to coordinate
4. **Work in feature branches** - Don't commit directly to main

### Recommended Workflow

```bash
# 1. Create a branch for your task
git checkout -b add-task/my_new_task

# 2. Create and develop your task
task new my_new_task
# ... edit files ...

# 3. Validate before committing
task validate tasks/my_new_task/

# 4. Commit and push
git add tasks/my_new_task/
git commit -m "Add my_new_task"
git push origin add-task/my_new_task

# 5. Open a PR for review
```

### Merging Tasks

Since tasks are in separate directories with unique names, merges are typically
conflict-free. The only potential conflict is in the compiled output files,
which should be regenerated after merging:

```bash
# After merging
task compile-all
```

## Task File Reference

### problem.yaml

```yaml
problem_id: "task_name"        # Unique identifier (use descriptive names!)
problem_name: "Human Name"     # Display name
domain: "your_domain"          # Domain/category for organization

description: |                 # Main problem description
  What this task is about...

io_spec: |                     # Input/output docstring
  """
  Parameters and returns...
  """

dependencies:                  # Python imports
  - import numpy as np
  - import scipy as sp

steps:                         # Ordered list of step files
  - 01_first_step
  - 02_second_step

background_main: |             # Optional overall background
  Scientific context...
```

### Step Files (steps/*.py)

Each step file contains:
1. **Module docstring** - Becomes `step_description_prompt`
2. **Function signature** - Becomes `function_header` + `return_line`
3. **Gold solution** - Named `_gold_{function_name}`
4. **Test cases** - `test_cases()` function returning test specs

### Background Files (background/*.md)

Optional Markdown files with scientific background for each step.
Named to match step files: `01_first_step.md` for `01_first_step.py`

## Compilation Output

Running `task compile` generates:
- `eval/data/problems.jsonl` - Task definitions
- `eval/data/test_data.h5` - Test target values (merged with existing)

These files are what the evaluation harness uses.

