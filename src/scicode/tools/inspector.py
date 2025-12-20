#!/usr/bin/env python3
"""
Task Structure Inspector

Unpacks and displays all components of a compiled task to help
understand the data structure when authoring new tasks.

Usage:
    inspect-task [problem_id]     # Inspect a specific task
    inspect-task                  # List available tasks
    
Examples:
    inspect-task my_task_name
"""

import json
import sys
from pathlib import Path
from textwrap import indent

import h5py
import numpy as np


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_data_paths():
    """Get paths to data files."""
    root = get_project_root()
    return {
        "h5": root / "eval" / "data" / "test_data.h5",
        "jsonl": root / "eval" / "data" / "problems.jsonl",
        "templates": root / "eval" / "data",
    }


def print_section(title: str, content: str = None, char: str = "="):
    """Print a formatted section header"""
    width = 80
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")
    if content:
        print(content)


def print_subsection(title: str, content: str = None):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")
    if content:
        print(content)


def truncate_str(s: str, max_len: int = 500) -> str:
    """Truncate long strings for display"""
    if len(s) > max_len:
        return s[:max_len] + f"\n... [truncated, {len(s)} total chars]"
    return s


def load_problem_from_jsonl(file_path: Path, problem_id: str = None):
    """Load problems from local JSONL file"""
    problems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            prob = json.loads(line.strip())
            if problem_id is None or prob['problem_id'] == problem_id:
                problems.append(prob)
    return problems


def explore_h5_structure(h5_file: Path, step_id: str = None, max_depth: int = 3):
    """Explore the structure of the H5 file"""
    if not h5_file.exists():
        return None
    
    structure = {}
    
    def recurse(group, path="", depth=0):
        if depth > max_depth:
            return "..."
        result = {}
        for key in group.keys():
            item = group[key]
            full_path = f"{path}/{key}" if path else key
            if isinstance(item, h5py.Group):
                result[key] = recurse(item, full_path, depth + 1)
            else:  # Dataset
                result[key] = {
                    "dtype": str(item.dtype),
                    "shape": item.shape,
                    "sample": repr(item[()]) if item.size < 10 else f"[{item.dtype} array, shape={item.shape}]"
                }
        return result
    
    with h5py.File(h5_file, 'r') as f:
        if step_id and step_id in f:
            structure[step_id] = recurse(f[step_id])
        else:
            # Just show top-level keys
            structure["top_level_keys"] = list(f.keys())[:20] + (["..."] if len(f.keys()) > 20 else [])
    
    return structure


def load_test_targets(h5_file: Path, step_id: str, num_tests: int):
    """Load the actual test target values from H5 file"""
    if not h5_file.exists():
        return None
    
    from scicode.parse.parse import process_hdf5_to_tuple
    try:
        return process_hdf5_to_tuple(step_id, num_tests, str(h5_file))
    except Exception as e:
        return f"Error loading targets: {e}"


def unpack_task(problem_data: dict):
    """Unpack and display all components of a single task"""
    
    paths = get_data_paths()
    
    # ========================================================================
    # 1. TOP-LEVEL PROBLEM STRUCTURE
    # ========================================================================
    print_section("1. TOP-LEVEL PROBLEM STRUCTURE")
    
    print("\nAll top-level fields in the problem:")
    for key in problem_data.keys():
        value = problem_data[key]
        if isinstance(value, str):
            print(f"  - {key}: (str, {len(value)} chars)")
        elif isinstance(value, list):
            print(f"  - {key}: (list, {len(value)} items)")
        else:
            print(f"  - {key}: ({type(value).__name__})")
    
    # ========================================================================
    # 2. PROBLEM METADATA
    # ========================================================================
    print_section("2. PROBLEM METADATA")
    
    print(f"\nproblem_id: {problem_data['problem_id']}")
    print(f"problem_name: {problem_data['problem_name']}")
    print(f"\nproblem_description_main:")
    print(indent(truncate_str(problem_data['problem_description_main'], 1000), "  "))
    
    print(f"\nproblem_io (Input/Output specification):")
    print(indent(truncate_str(problem_data['problem_io'], 1500), "  "))
    
    if problem_data.get('problem_background_main'):
        print(f"\nproblem_background_main:")
        print(indent(truncate_str(problem_data['problem_background_main'], 500), "  "))
    else:
        print("\nproblem_background_main: (empty)")
    
    # ========================================================================
    # 3. DEPENDENCIES
    # ========================================================================
    print_section("3. REQUIRED DEPENDENCIES")
    print(f"\n{problem_data['required_dependencies']}")
    
    # ========================================================================
    # 4. SUB-STEPS STRUCTURE
    # ========================================================================
    print_section("4. SUB-STEPS (The heart of the task)")
    
    sub_steps = problem_data['sub_steps']
    print(f"\nTotal sub-steps: {len(sub_steps)}")
    
    for i, step in enumerate(sub_steps):
        print_subsection(f"Sub-step {i+1}: {step['step_number']}")
        
        print("\nFields in this sub-step:")
        for key in step.keys():
            value = step[key]
            if isinstance(value, str):
                print(f"    - {key}: (str, {len(value)} chars)")
            elif isinstance(value, list):
                print(f"    - {key}: (list, {len(value)} items)")
            else:
                print(f"    - {key}: ({type(value).__name__})")
        
        print(f"\n  step_number: {step['step_number']}")
        print(f"\n  step_description_prompt:")
        print(indent(truncate_str(step['step_description_prompt'], 600), "    "))
        
        print(f"\n  function_header:")
        print(indent(step['function_header'], "    "))
        
        print(f"\n  return_line:")
        print(f"    {step['return_line']}")
        
        if step.get('step_background'):
            print(f"\n  step_background (context):")
            print(indent(truncate_str(step['step_background'], 600), "    "))
        
        # Test cases
        print(f"\n  test_cases ({len(step['test_cases'])} tests):")
        for j, test in enumerate(step['test_cases']):
            print(f"\n    Test {j+1}:")
            print(indent(truncate_str(test, 300), "      "))
        
        # Ground truth (if you have it)
        if 'ground_truth_code' in step:
            print(f"\n  ground_truth_code:")
            print(indent(truncate_str(step.get('ground_truth_code', 'N/A'), 400), "    "))
        
        print("\n" + "-" * 40)
    
    # ========================================================================
    # 5. GENERAL TESTS (End-to-end tests)
    # ========================================================================
    print_section("5. GENERAL TESTS (End-to-end validation)")
    
    general_tests = problem_data.get('general_tests', [])
    print(f"\nTotal general tests: {len(general_tests)}")
    
    for i, test in enumerate(general_tests):
        print(f"\n  General Test {i+1}:")
        print(indent(truncate_str(test, 1500), "    "))
    
    # ========================================================================
    # 6. H5PY TEST DATA STRUCTURE
    # ========================================================================
    print_section("6. H5PY TEST DATA STRUCTURE")
    
    h5_file = paths["h5"]
    if h5_file.exists():
        print(f"\nH5 file location: {h5_file}")
        
        # Show structure for first sub-step
        first_step_id = sub_steps[0]['step_number']
        print(f"\nExploring structure for step '{first_step_id}':")
        structure = explore_h5_structure(h5_file, first_step_id)
        print(json.dumps(structure, indent=2, default=str)[:2000])
        
        # Load actual target values
        print(f"\n\nLoading actual target values for step '{first_step_id}':")
        num_tests = len(sub_steps[0]['test_cases'])
        targets = load_test_targets(h5_file, first_step_id, num_tests)
        if targets:
            for i, target in enumerate(targets):
                print(f"\n  Target {i+1}:")
                target_str = repr(target)
                print(indent(truncate_str(target_str, 300), "    "))
    else:
        print(f"\nH5 file not found at: {h5_file}")
    
    # ========================================================================
    # 7. PROMPT GENERATION EXAMPLE
    # ========================================================================
    print_section("7. PROMPT GENERATION EXAMPLE")
    
    # Load template
    template_path = paths["templates"] / "background_comment_template.txt"
    if template_path.exists():
        template = template_path.read_text()
        print("\nTemplate file: background_comment_template.txt")
        print("\nTemplate structure:")
        print(indent(template, "  "))
        
        # Show what a prompt would look like for step 1
        print("\n\nExample prompt for step 1 (without background):")
        step1 = sub_steps[0]
        problem_steps_str = ""  # Empty for step 1
        next_step_str = f"{step1['step_description_prompt']}\n\n{step1['function_header']}\n\n{step1['return_line']}"
        
        example_prompt = template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=problem_data['required_dependencies']
        )
        print(indent(truncate_str(example_prompt, 2000), "  "))
    
    # ========================================================================
    # 8. OUTPUT FILES GENERATED DURING EVALUATION
    # ========================================================================
    print_section("8. FILES GENERATED DURING EVALUATION")
    
    print("""
During evaluation, the following files are created:

Directory Structure:
    {output_dir}/
    └── {model_name}/
        ├── generated_code/
        │   └── {with_background|without_background}/
        │       ├── {problem_id}.1.py   # Generated code for step 1
        │       ├── {problem_id}.2.py   # Generated code for step 2
        │       └── ...
        ├── prompt/
        │   └── {with_background|without_background}/
        │       ├── {problem_id}.1.txt  # Prompt sent for step 1
        │       ├── {problem_id}.2.txt  # Prompt sent for step 2
        │       └── ...
        └── evaluation_logs/
            └── {with_background|without_background}/
                ├── {step_number}.log   # 'pass', 'fail', or 'time out'
                └── ...
""")
    
    # ========================================================================
    # 9. SCHEMA SUMMARY FOR NEW TASKS
    # ========================================================================
    print_section("9. SCHEMA SUMMARY FOR ADDING NEW TASKS")
    
    print("""
To add a new task, you need to provide a JSON object with this structure:

{
    "problem_id": "unique_id",           # String identifier
    "problem_name": "Descriptive_Name",  # Human-readable name
    "problem_description_main": "...",   # Main problem description
    "problem_io": "...",                 # Input/output docstring
    "problem_background_main": "",       # Optional background
    "required_dependencies": "import ...",  # Python imports needed
    
    "sub_steps": [                       # List of sub-problems
        {
            "step_number": "task_id.1",  # Unique step ID (problem_id.step_num)
            "step_description_prompt": "...",  # What to implement
            "function_header": "def func(...):\\n    '''docstring'''",
            "return_line": "    return result",
            "step_background": "...",    # Context (optional)
            "test_cases": [              # Assertion-based tests
                "x = 5\\nassert func(x) == target",
                "..."
            ],
            "ground_truth_code": "..."   # Reference implementation (optional)
        },
        ...
    ],
    
    "general_tests": [                   # End-to-end validation tests
        "# Full integration test code\\nassert final_result == target"
    ]
}

Additionally, you need to add test target data to the H5 file:
    - Path: {step_number}/test{N}/var{M}
    - Contains expected outputs that 'target' refers to in test_cases
""")
    
    # ========================================================================
    # 10. RAW JSON DUMP
    # ========================================================================
    print_section("10. RAW JSON (First 5000 chars)")
    
    raw_json = json.dumps(problem_data, indent=2)
    print(truncate_str(raw_json, 5000))


def main():
    """Main entry point."""
    # Parse command line args
    problem_id = None
    
    for arg in sys.argv[1:]:
        if arg == "--help" or arg == "-h":
            print(__doc__)
            return 0
        else:
            problem_id = arg
    
    paths = get_data_paths()
    jsonl_path = paths["jsonl"]
    
    # Load from local JSONL
    if not jsonl_path.exists():
        print(f"\nError: No compiled tasks found at {jsonl_path}")
        print("Run 'task compile-all' first.")
        return 1
    
    all_problems = load_problem_from_jsonl(jsonl_path)
    
    if not problem_id:
        # List available tasks
        print("\nAvailable tasks:")
        for p in all_problems:
            print(f"  - {p['problem_id']}: {p['problem_name']}")
        print(f"\nUsage: inspect-task <problem_id>")
        return 0
    
    print(f"\n{'#' * 80}")
    print(f"# Task Inspector - {problem_id}")
    print(f"{'#' * 80}")
    
    # Find the problem
    problems = [p for p in all_problems if p['problem_id'] == problem_id]
    
    if not problems:
        print(f"\nError: Could not find problem '{problem_id}'")
        print(f"Available tasks: {[p['problem_id'] for p in all_problems]}")
        return 1
    
    # Unpack the matching problem
    unpack_task(problems[0])
    
    print("\n" + "=" * 80)
    print(" DONE! This shows the compiled structure of your task.")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

