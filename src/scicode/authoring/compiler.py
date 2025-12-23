"""
Task Compiler

Compiles task definitions from the human-friendly YAML/Python format
into the machine-readable JSON + H5 format used by SciCode evaluation.
"""

import ast
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .validator import TaskValidator


class TaskCompiler:
    """Compiles task directories into SciCode evaluation format."""
    
    def __init__(self, task_dir: Path, output_dir: Path = None):
        """
        Initialize compiler for a task directory.
        
        Args:
            task_dir: Path to the task directory (e.g., tasks/physics_pbc/)
            output_dir: Output directory for compiled files (default: eval/data/)
        """
        self.task_dir = Path(task_dir)
        self.output_dir = output_dir or Path("eval/data")
        self.validator = TaskValidator(task_dir)
        
        # Loaded data
        self.problem_yaml = None
        self.steps = []
        self.backgrounds = {}
        
    def load(self) -> "TaskCompiler":
        """Load all task files."""
        self._load_problem_yaml()
        self._load_steps()
        self._load_backgrounds()
        return self
        
    def _load_problem_yaml(self):
        """Load problem.yaml configuration."""
        yaml_path = self.task_dir / "problem.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"problem.yaml not found in {self.task_dir}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.problem_yaml = yaml.safe_load(f)
            
    def _load_steps(self):
        """Load all step files."""
        steps_dir = self.task_dir / "steps"
        if not steps_dir.exists():
            raise FileNotFoundError(f"steps/ directory not found in {self.task_dir}")
        
        self.steps = []
        for step_name in self.problem_yaml.get("steps", []):
            step_path = steps_dir / f"{step_name}.py"
            if not step_path.exists():
                raise FileNotFoundError(f"Step file not found: {step_path}")
            
            step_data = self._parse_step_file(step_path, step_name)
            self.steps.append(step_data)
            
    def _parse_step_file(self, step_path: Path, step_name: str) -> dict:
        """Parse a step Python file into structured data."""
        with open(step_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        # Extract module docstring
        module_docstring = ast.get_docstring(tree) or ""
        
        # Find the main function (not _gold_ prefixed)
        main_func = None
        gold_func = None
        test_cases_func = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("_gold_"):
                    gold_func = node
                elif node.name == "test_cases":
                    test_cases_func = node
                elif not node.name.startswith("_"):
                    main_func = node
        
        if main_func is None:
            raise ValueError(f"No main function found in {step_path}")
        
        # Extract function header (signature + docstring)
        func_header = self._extract_function_header(content, main_func)
        
        # Extract return line
        return_line = self._extract_return_line(main_func)
        
        # Extract and run test cases
        test_cases = self._extract_test_cases(step_path, content)
        
        # Extract gold solution code
        gold_code = ""
        if gold_func:
            gold_code = ast.unparse(gold_func)
        
        return {
            "step_name": step_name,
            "step_path": step_path,
            "description": module_docstring.strip(),
            "function_name": main_func.name,
            "function_header": func_header,
            "return_line": return_line,
            "test_cases": test_cases,
            "gold_function": f"_gold_{main_func.name}" if gold_func else None,
            "gold_code": gold_code,
            "source_content": content,
        }
    
    def _extract_function_header(self, content: str, func_node: ast.FunctionDef) -> str:
        """Extract function signature and docstring."""
        lines = content.split('\n')
        
        # Get the def line
        start_line = func_node.lineno - 1
        end_line = start_line
        
        # Find where the docstring ends (or function body starts)
        docstring = ast.get_docstring(func_node)
        if docstring:
            # Find the closing quotes of the docstring
            for i, line in enumerate(lines[start_line:], start=start_line):
                if "'''" in line or '"""' in line:
                    # Check if this is the end of docstring
                    if i > start_line:
                        # Count quotes to find the closing
                        quote_count = line.count("'''") + line.count('"""')
                        if quote_count >= 1:
                            end_line = i
                            # Check if it's a single-line or end of multi-line
                            if i == start_line or (line.strip().endswith("'''") or line.strip().endswith('"""')):
                                break
        else:
            # No docstring, just get the def line
            end_line = start_line
            
        # Build the header
        header_lines = []
        in_docstring = False
        docstring_char = None
        
        for i in range(start_line, min(end_line + 10, len(lines))):
            line = lines[i]
            header_lines.append(line)
            
            # Track docstring boundaries
            if not in_docstring:
                if "'''" in line:
                    in_docstring = True
                    docstring_char = "'''"
                    if line.count("'''") >= 2:  # Single line docstring
                        in_docstring = False
                        break
                elif '"""' in line:
                    in_docstring = True
                    docstring_char = '"""'
                    if line.count('"""') >= 2:  # Single line docstring
                        in_docstring = False
                        break
            else:
                if docstring_char in line:
                    in_docstring = False
                    break
        
        return '\n'.join(header_lines)
    
    def _extract_return_line(self, func_node: ast.FunctionDef) -> str:
        """Extract the return statement from the function."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                if node.value:
                    return_var = ast.unparse(node.value)
                    return f"    return {return_var}"
        return "    return result"
    
    def _extract_test_cases(self, step_path: Path, content: str) -> list:
        """Extract and process test cases from the step file."""
        # Import the module to get test_cases()
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("step_module", step_path)
        module = importlib.util.module_from_spec(spec)
        
        # Add numpy to module namespace
        module.__dict__['np'] = np
        module.__dict__['numpy'] = np
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Warning: Could not execute module {step_path}: {e}")
            return []
        
        if not hasattr(module, 'test_cases'):
            return []
        
        return module.test_cases()
    
    def _load_backgrounds(self):
        """Load background markdown files."""
        bg_dir = self.task_dir / "background"
        if not bg_dir.exists():
            return
        
        for step_data in self.steps:
            bg_path = bg_dir / f"{step_data['step_name']}.md"
            if bg_path.exists():
                with open(bg_path, 'r', encoding='utf-8') as f:
                    self.backgrounds[step_data['step_name']] = f.read()
                    
    
    def generate_targets(self) -> dict:
        """
        Run gold solutions to generate test targets.
        
        Returns:
            dict mapping step_id -> list of target values
        """
        targets = {}
        problem_id = self.problem_yaml["problem_id"]
        
        # Build execution context with all gold functions
        exec_context = {"np": np, "numpy": np}
        
        for step_data in self.steps:
            step_path = step_data["step_path"]
            
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location("step", step_path)
            module = importlib.util.module_from_spec(spec)
            module.__dict__.update(exec_context)
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"Error loading {step_path}: {e}")
                continue
            
            # Add all functions to context for subsequent steps
            for name in dir(module):
                if callable(getattr(module, name)) and not name.startswith('__'):
                    exec_context[name] = getattr(module, name)
            
            # Generate targets for each test case
            step_num = step_data["step_name"].split("_")[0]  # e.g., "01" from "01_wrap"
            step_id = f"{problem_id}.{int(step_num)}"
            
            step_targets = []
            for tc in step_data.get("test_cases", []):
                try:
                    # Execute setup
                    local_ctx = exec_context.copy()
                    exec(tc["setup"], local_ctx)
                    
                    # Execute gold call
                    target = eval(tc["gold_call"], local_ctx)
                    step_targets.append(target)
                except Exception as e:
                    print(f"Error generating target for {step_id}: {e}")
                    step_targets.append(None)
            
            targets[step_id] = step_targets
        
        return targets
    
    def to_json(self) -> dict:
        """Convert to SciCode JSON format."""
        problem_id = self.problem_yaml["problem_id"]
        
        # Build dependencies string
        deps = self.problem_yaml.get("dependencies", [])
        dependencies = "\n".join(deps)
        
        # Build sub_steps
        sub_steps = []
        for i, step_data in enumerate(self.steps, 1):
            step_num = step_data["step_name"].split("_")[0]
            step_id = f"{problem_id}.{int(step_num)}"
            
            # Build test case strings
            test_case_strs = []
            func_name = step_data["function_name"]
            
            for tc in step_data.get("test_cases", []):
                setup = tc["setup"].strip()
                call = tc["call"]
                # Generate assertion that uses 'target'
                test_str = f"{setup}\nassert np.allclose({call}, target)"
                test_case_strs.append(test_str)
            
            # Convert gold solution to the expected format (rename function)
            gold_code = step_data.get("gold_code", "")
            if gold_code and step_data.get("gold_function"):
                # Rename _gold_func to func in the gold code
                gold_func_name = step_data["gold_function"]
                main_func_name = step_data["function_name"]
                gold_code = gold_code.replace(f"def {gold_func_name}", f"def {main_func_name}")
            
            sub_step = {
                "step_number": step_id,
                "step_description_prompt": step_data["description"],
                "function_header": step_data["function_header"],
                "return_line": step_data["return_line"],
                "test_cases": test_case_strs,
                "step_background": self.backgrounds.get(step_data["step_name"], ""),
                "ground_truth_code": gold_code,
            }
            sub_steps.append(sub_step)
        
        # Build general tests from the last step's test cases
        # In SciCode, general_tests are the final subproblem's tests since
        # the last step implicitly validates all previous steps worked correctly
        general_tests = []
        if self.steps:
            last_step = self.steps[-1]
            for tc in last_step.get("test_cases", []):
                setup = tc["setup"].strip()
                call = tc["call"]
                test_str = f"{setup}\nassert np.allclose({call}, target)"
                general_tests.append(test_str)
        
        return {
            "problem_id": problem_id,
            "problem_name": self.problem_yaml.get("problem_name", ""),
            "domain": self.problem_yaml.get("domain", ""),
            "subdomain": self.problem_yaml.get("subdomain", ""),
            "problem_description_main": self.problem_yaml.get("description", ""),
            "problem_io": self.problem_yaml.get("io_spec", ""),
            "required_dependencies": dependencies,
            "sub_steps": sub_steps,
            "general_tests": general_tests,
            "problem_background_main": self.problem_yaml.get("background_main", ""),
        }
    
    def compile(self, merge_existing: bool = True) -> Path:
        """
        Compile the task to JSON and H5 files.
        
        Args:
            merge_existing: If True, merge with existing files instead of overwriting
            
        Returns:
            Path to the output directory
        """
        # Validate first
        errors = self.validator.validate()
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  - {e}")
            raise ValueError("Task validation failed")
        
        # Load all data
        self.load()
        
        # Generate JSON
        json_data = self.to_json()
        
        # Generate targets
        targets = self.generate_targets()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write JSON
        jsonl_path = self.output_dir / "problems.jsonl"
        existing_problems = {}
        
        if merge_existing and jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    prob = json.loads(line.strip())
                    existing_problems[prob["problem_id"]] = prob
        
        existing_problems[json_data["problem_id"]] = json_data
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for prob in existing_problems.values():
                f.write(json.dumps(prob) + '\n')
        
        print(f"Written: {jsonl_path}")
        
        # Write H5 targets
        h5_path = self.output_dir / "test_data.h5"
        
        with h5py.File(h5_path, 'a') as h5f:
            for step_id, step_targets in targets.items():
                for i, target in enumerate(step_targets, 1):
                    if target is None:
                        continue
                    
                    group_path = f"{step_id}/test{i}"
                    
                    # Remove existing group if present
                    if group_path in h5f:
                        del h5f[group_path]
                    
                    group = h5f.create_group(group_path)
                    
                    # Store target value
                    if isinstance(target, np.ndarray):
                        group.create_dataset("var1", data=target)
                    elif isinstance(target, (int, float)):
                        group.create_dataset("var1", data=target)
                    elif isinstance(target, str):
                        group.create_dataset("var1", data=target.encode('utf-8'))
                    elif isinstance(target, (list, tuple)):
                        try:
                            group.create_dataset("var1", data=np.array(target))
                        except:
                            # Complex nested structure
                            for j, item in enumerate(target):
                                group.create_dataset(f"var{j+1}", data=np.array(item))
                    else:
                        group.create_dataset("var1", data=str(target).encode('utf-8'))
        
        print(f"Written: {h5_path}")
        
        return self.output_dir


def create_task_from_template(task_name: str, tasks_dir: Path = None) -> Path:
    """
    Create a new task directory from the template.
    
    Args:
        task_name: Name for the new task (e.g., "physics_lennard_jones")
        tasks_dir: Base tasks directory (default: tasks/)
        
    Returns:
        Path to the created task directory
    """
    tasks_dir = tasks_dir or Path("tasks")
    template_dir = tasks_dir / "_template"
    new_task_dir = tasks_dir / task_name
    
    if not template_dir.exists():
        raise FileNotFoundError(f"Template not found at {template_dir}")
    
    if new_task_dir.exists():
        raise FileExistsError(f"Task directory already exists: {new_task_dir}")
    
    # Copy template
    shutil.copytree(template_dir, new_task_dir)
    
    # Update problem.yaml with the task name
    yaml_path = new_task_dir / "problem.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace placeholder with actual task name
    content = content.replace('problem_id: "DOMAIN_SHORT_NAME"', f'problem_id: "{task_name}"')
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created new task: {new_task_dir}")
    print(f"\nNext steps:")
    print(f"  1. Edit {yaml_path}")
    print(f"  2. Edit/add step files in {new_task_dir}/steps/")
    print(f"  3. Optionally add background in {new_task_dir}/background/")
    print(f"  4. Compile: python compile_tasks.py compile {new_task_dir}")
    
    return new_task_dir

