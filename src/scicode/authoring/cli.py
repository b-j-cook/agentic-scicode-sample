#!/usr/bin/env python3
"""
Task CLI

Compile tasks from the human-friendly YAML/Python format into the 
machine-readable JSON + H5 format used for evaluation.

Usage:
    task new <task_name>          Create new task from template
    task validate <task_dir>      Validate a task
    task compile <task_dir>       Compile a task
    task compile-all              Compile all tasks
    task test <task_dir>          Compile and test with eval harness
    task list                     List all tasks

Examples:
    task new lennard_jones_potential
    task validate tasks/lennard_jones_potential/
    task compile tasks/lennard_jones_potential/
    task test tasks/lennard_jones_potential/
"""

import argparse
import subprocess
import sys
from pathlib import Path

from scicode.authoring.compiler import TaskCompiler, create_task_from_template
from scicode.authoring.validator import TaskValidator


def get_project_root():
    """Get the project root directory."""
    # Navigate up from src/scicode/authoring/cli.py to project root
    return Path(__file__).parent.parent.parent.parent


def cmd_new(args):
    """Create a new task from template."""
    task_name = args.task_name
    tasks_dir = get_project_root() / args.tasks_dir
    
    # Validate task name
    if task_name.isdigit():
        print("Error: Task name should not be purely numeric.")
        print("Use descriptive names like 'lennard_jones_potential' to avoid merge conflicts.")
        return 1
    
    if " " in task_name:
        print("Error: Task name should not contain spaces.")
        return 1
    
    try:
        path = create_task_from_template(task_name, tasks_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_validate(args):
    """Validate a task."""
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = get_project_root() / task_dir
    
    if not task_dir.exists():
        print(f"Error: Task directory not found: {task_dir}")
        return 1
    
    validator = TaskValidator(task_dir)
    validator.print_report()
    
    return 0 if validator.is_valid() else 1


def cmd_compile(args):
    """Compile a task."""
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = get_project_root() / task_dir
    
    output_dir = Path(args.output_dir) if args.output_dir else get_project_root() / "eval/data"
    
    if not task_dir.exists():
        print(f"Error: Task directory not found: {task_dir}")
        return 1
    
    # Skip template directory
    if task_dir.name.startswith("_"):
        print(f"Skipping template directory: {task_dir}")
        return 0
    
    try:
        compiler = TaskCompiler(task_dir, output_dir)
        compiler.compile(merge_existing=not args.overwrite)
        print(f"\n✓ Successfully compiled: {task_dir.name}")
        return 0
    except Exception as e:
        print(f"\n✗ Compilation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_compile_all(args):
    """Compile all tasks."""
    tasks_dir = get_project_root() / args.tasks_dir
    output_dir = Path(args.output_dir) if args.output_dir else get_project_root() / "eval/data"
    
    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found: {tasks_dir}")
        return 1
    
    # Find all task directories (exclude _template and other _ prefixed)
    task_dirs = [
        d for d in sorted(tasks_dir.iterdir()) 
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]
    
    if not task_dirs:
        print("No tasks found to compile.")
        return 0
    
    print(f"Found {len(task_dirs)} task(s) to compile\n")
    
    success = 0
    failed = 0
    
    for task_dir in task_dirs:
        print(f"Compiling: {task_dir.name}...")
        try:
            compiler = TaskCompiler(task_dir, output_dir)
            compiler.compile(merge_existing=True)
            print(f"  ✓ Success")
            success += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
    
    print(f"\nSummary: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_test(args):
    """Compile and test a task with the eval harness."""
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = get_project_root() / task_dir
    
    if not task_dir.exists():
        print(f"Error: Task directory not found: {task_dir}")
        return 1
    
    # First compile
    print("=" * 60)
    print("Step 1: Compiling task...")
    print("=" * 60)
    
    try:
        output_dir = get_project_root() / "eval/data"
        compiler = TaskCompiler(task_dir, output_dir)
        compiler.compile(merge_existing=True)
        
        problem_id = compiler.problem_yaml["problem_id"]
        print(f"✓ Compiled task: {problem_id}")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return 1
    
    # Then run eval harness with gold mode
    print("\n" + "=" * 60)
    print("Step 2: Running evaluation with gold solutions...")
    print("=" * 60)
    
    # Check if inspect is available
    try:
        import inspect_ai
    except ImportError:
        print("Error: inspect_ai not installed. Run: pip install inspect-ai")
        return 1
    
    # Run eval with gold mode on this specific problem
    project_root = get_project_root()
    cmd = [
        "inspect", "eval", str(project_root / "eval/inspect_ai/scicode.py"),
        "--model", "openai/gpt-4o",  # Model doesn't matter in gold mode
        "-T", "mode=gold",
        "-T", f"problems={problem_id}",
        "-T", "output_dir=./tmp_test",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✓ Task passed evaluation with gold solutions!")
    else:
        print("\n✗ Task failed evaluation")
    
    return result.returncode


def cmd_list(args):
    """List all tasks."""
    tasks_dir = get_project_root() / args.tasks_dir
    
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        return 1
    
    task_dirs = [
        d for d in sorted(tasks_dir.iterdir())
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]
    
    if not task_dirs:
        print("No tasks found.")
        print(f"\nTo create a new task: task new <task_name>")
        return 0
    
    print(f"Found {len(task_dirs)} task(s):\n")
    
    for task_dir in task_dirs:
        yaml_path = task_dir / "problem.yaml"
        if yaml_path.exists():
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            problem_id = data.get("problem_id", "?")
            name = data.get("problem_name", "?")
            domain = data.get("domain", "?")
            steps = len(data.get("steps", []))
            
            print(f"  {task_dir.name}/")
            print(f"    ID: {problem_id}")
            print(f"    Name: {name}")
            print(f"    Domain: {domain}")
            print(f"    Steps: {steps}")
            print()
        else:
            print(f"  {task_dir.name}/ (missing problem.yaml)")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Task Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # new command
    new_parser = subparsers.add_parser("new", help="Create new task from template")
    new_parser.add_argument("task_name", help="Name for the new task (e.g., lennard_jones_potential)")
    new_parser.add_argument("--tasks-dir", default="tasks", help="Tasks directory")
    
    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a task")
    validate_parser.add_argument("task_dir", help="Path to task directory")
    
    # compile command
    compile_parser = subparsers.add_parser("compile", help="Compile a task")
    compile_parser.add_argument("task_dir", help="Path to task directory")
    compile_parser.add_argument("--output-dir", "-o", help="Output directory")
    compile_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    compile_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # compile-all command
    compile_all_parser = subparsers.add_parser("compile-all", help="Compile all tasks")
    compile_all_parser.add_argument("--tasks-dir", default="tasks", help="Tasks directory")
    compile_all_parser.add_argument("--output-dir", "-o", help="Output directory")
    
    # test command
    test_parser = subparsers.add_parser("test", help="Compile and test with eval harness")
    test_parser.add_argument("task_dir", help="Path to task directory")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List all tasks")
    list_parser.add_argument("--tasks-dir", default="tasks", help="Tasks directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "new": cmd_new,
        "validate": cmd_validate,
        "compile": cmd_compile,
        "compile-all": cmd_compile_all,
        "test": cmd_test,
        "list": cmd_list,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
