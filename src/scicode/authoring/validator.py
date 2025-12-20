"""
Task Validator

Validates task definitions before compilation to catch errors early.
"""

import ast
from pathlib import Path
from typing import List, Optional

import yaml


class TaskValidator:
    """Validates task directory structure and content."""
    
    REQUIRED_YAML_FIELDS = [
        "problem_id",
        "problem_name", 
        "description",
        "io_spec",       # Maps to problem_io in original - always has content
        "dependencies",
        "steps",
    ]
    
    VALID_DOMAINS = [
        "physics",
        "chemistry", 
        "biology",
        "math",
        "materials",
        "other",
    ]
    
    def __init__(self, task_dir: Path):
        """
        Initialize validator for a task directory.
        
        Args:
            task_dir: Path to the task directory
        """
        self.task_dir = Path(task_dir)
        self.errors: List[str] = []
        self._is_template = self.task_dir.name == "_template"
        
    def validate(self) -> List[str]:
        """
        Run all validation checks.
        
        Returns:
            List of error messages (empty if valid)
        """
        self.errors = []
        
        # Check directory structure
        self._validate_directory_structure()
        
        # Check problem.yaml
        if (self.task_dir / "problem.yaml").exists():
            self._validate_problem_yaml()
        
        # Check step files
        self._validate_step_files()
        
        return self.errors
    
    def _validate_directory_structure(self):
        """Check required directories and files exist."""
        if not self.task_dir.exists():
            self.errors.append(f"Task directory does not exist: {self.task_dir}")
            return
        
        if not (self.task_dir / "problem.yaml").exists():
            self.errors.append("Missing required file: problem.yaml")
            
        if not (self.task_dir / "steps").exists():
            self.errors.append("Missing required directory: steps/")
        elif not list((self.task_dir / "steps").glob("*.py")):
            self.errors.append("No step files found in steps/")
    
    def _validate_problem_yaml(self):
        """Validate problem.yaml content."""
        yaml_path = self.task_dir / "problem.yaml"
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in problem.yaml: {e}")
            return
        
        if data is None:
            self.errors.append("problem.yaml is empty")
            return
        
        # Check required fields
        for field in self.REQUIRED_YAML_FIELDS:
            if field not in data:
                self.errors.append(f"Missing required field in problem.yaml: {field}")
            elif not data[field]:
                self.errors.append(f"Empty required field in problem.yaml: {field}")
        
        # Validate problem_id format (skip for template)
        problem_id = data.get("problem_id", "")
        if problem_id and not self._is_template:
            if problem_id.isdigit():
                self.errors.append(
                    f"problem_id '{problem_id}' should not be numeric. "
                    "Use descriptive names like 'lennard_jones_potential' to avoid merge conflicts."
                )
            if " " in problem_id:
                self.errors.append(f"problem_id '{problem_id}' should not contain spaces")
            if not problem_id.replace("_", "").replace("-", "").isalnum():
                self.errors.append(
                    f"problem_id '{problem_id}' should only contain letters, numbers, underscores, and hyphens"
                )
        
        # Validate domain (skip for template)
        domain = data.get("domain", "")
        if domain and domain not in self.VALID_DOMAINS and not self._is_template:
            self.errors.append(
                f"Invalid domain '{domain}'. Valid options: {', '.join(self.VALID_DOMAINS)}"
            )
        
        # Check that all listed steps have corresponding files
        steps = data.get("steps", [])
        steps_dir = self.task_dir / "steps"
        
        if steps_dir.exists():
            for step_name in steps:
                step_path = steps_dir / f"{step_name}.py"
                if not step_path.exists():
                    self.errors.append(f"Step file not found: {step_path}")
    
    def _validate_step_files(self):
        """Validate each step file."""
        steps_dir = self.task_dir / "steps"
        if not steps_dir.exists():
            return
        
        for step_path in sorted(steps_dir.glob("*.py")):
            self._validate_step_file(step_path)
    
    def _validate_step_file(self, step_path: Path):
        """Validate a single step file."""
        try:
            with open(step_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read {step_path}: {e}")
            return
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {step_path}: {e}")
            return
        
        # Check for module docstring
        if not ast.get_docstring(tree):
            self.errors.append(f"{step_path}: Missing module docstring (step description)")
        
        # Find functions
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
        
        # Find the main function (not _gold_, not test_cases, not _*)
        main_funcs = [
            name for name in functions.keys() 
            if not name.startswith('_') and name != 'test_cases'
        ]
        
        if not main_funcs:
            self.errors.append(f"{step_path}: No main function found")
            return
        
        if len(main_funcs) > 1:
            self.errors.append(
                f"{step_path}: Multiple main functions found: {main_funcs}. "
                "Only one non-underscore-prefixed function should be defined."
            )
        
        main_func_name = main_funcs[0]
        main_func = functions[main_func_name]
        
        # Check for docstring in main function
        if not ast.get_docstring(main_func):
            self.errors.append(f"{step_path}: Main function '{main_func_name}' missing docstring")
        
        # Check for gold solution
        gold_name = f"_gold_{main_func_name}"
        if gold_name not in functions:
            self.errors.append(f"{step_path}: Missing gold solution '{gold_name}'")
        
        # Check for test_cases function
        if 'test_cases' not in functions:
            self.errors.append(f"{step_path}: Missing test_cases() function")
        
        # Check for return statement in main function
        has_return = False
        for node in ast.walk(main_func):
            if isinstance(node, ast.Return):
                has_return = True
                break
        
        if not has_return:
            self.errors.append(f"{step_path}: Main function '{main_func_name}' has no return statement")
    
    def is_valid(self) -> bool:
        """Check if the task is valid."""
        return len(self.validate()) == 0
    
    def print_report(self):
        """Print validation report."""
        errors = self.validate()
        
        if not errors:
            if self._is_template:
                print(f"✓ Template '{self.task_dir.name}' structure is valid")
            else:
                print(f"✓ Task '{self.task_dir.name}' is valid")
            return
        
        print(f"✗ Task '{self.task_dir.name}' has {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")

