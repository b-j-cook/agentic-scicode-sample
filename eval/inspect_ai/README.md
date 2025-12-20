# Evaluation with inspect_ai

Run evaluations on your authored tasks using the [inspect_ai](https://inspect.ai-safety-institute.org.uk/) framework.

## Setup

### 1. Install Dependencies

```bash
# From project root
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Set API Keys

Set the environment variable for your model provider:

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

See [inspect_ai documentation](https://inspect.ai-safety-institute.org.uk/#getting-started) for other providers.

## Running Evaluations

### Evaluate Custom Tasks

```bash
cd eval/inspect_ai

# Evaluate a specific task
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=your_task_id \
    --model openai/gpt-4o \
    --temperature 0

# Evaluate multiple tasks
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=task1,task2,task3 \
    --model anthropic/claude-sonnet-4-5

# Evaluate all compiled tasks
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    --model openai/gpt-4o
```

### With Scientific Background

```bash
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T with_background=True \
    --model anthropic/claude-sonnet-4-5
```

## Command Line Options

### Task Selection

| Option | Description | Example |
|--------|-------------|---------|
| `-T local_jsonl=PATH` | Path to compiled problems.jsonl | `../data/problems.jsonl` |
| `-T problems=IDS` | Comma-separated task IDs | `task1,task2` |

### Evaluation Modes

| Option | Description |
|--------|-------------|
| `-T mode=normal` | Standard evaluation (default) |
| `-T mode=gold` | Test with gold solutions (sanity check) |
| `-T mode=dummy` | No LLM calls, dummy outputs |

### Output Control

| Option | Description | Default |
|--------|-------------|---------|
| `-T output_dir=PATH` | Generated code output | `./tmp` |
| `-T with_background=BOOL` | Include scientific background | `False` |
| `-T force=BOOL` | Clear existing results | `True` |

### Model Options

| Option | Description |
|--------|-------------|
| `--model MODEL` | Model identifier (e.g., `openai/gpt-4o`) |
| `--temperature FLOAT` | Sampling temperature |
| `--max-connections N` | Max concurrent API connections |
| `--limit N` | Limit number of samples |

## Examples by Provider

### Google Gemini

```bash
export GOOGLE_API_KEY=your-key

inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=your_task_id \
    --model google/gemini-3-pro-preview \
    --temperature 0
```

### OpenAI

```bash
export OPENAI_API_KEY=your-key

inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=your_task_id \
    --model openai/gpt-5.2 \
    --temperature 0
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=your-key

inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=your_task_id \
    --model anthropic/claude-sonnet-4-5 \
    --temperature 0
```

### Together AI

```bash
export TOGETHER_API_KEY=your-key

inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=your_task_id \
    --model together/deepseek-ai/DeepSeek-V3 \
    --max-connections 2 \
    --max-tokens 32784
```

## Testing Your Task

### Verify with Gold Solutions

Before evaluating with a real model, verify your task works:

```bash
inspect eval scicode.py \
    -T local_jsonl=../data/problems.jsonl \
    -T problems=your_task_id \
    -T mode=gold \
    --model mockllm/model
```

Expected output: `Problem Correctness/mean: 1.0`

## How Evaluation Works

1. **Load Task**: Read problem definition from `problems.jsonl`
2. **Sequential Steps**: For each sub-step:
   - Generate prompt with previous code context
   - Call LLM to generate solution
   - Extract Python code from response
   - Save to disk
3. **Run Tests**: Execute generated code against test cases
4. **Score**: 
   - Sub-problem passes if all its test cases pass
   - Main problem passes if all sub-problems pass

## Output Files

After evaluation, find results in your output directory:

```
{output_dir}/{model_name}/
├── generated_code/
│   └── {with|without}_background/
│       ├── {task_id}.1.py    # Step 1 generated code
│       ├── {task_id}.2.py    # Step 2 generated code
│       └── ...
├── prompt/
│   └── {with|without}_background/
│       ├── {task_id}.1.txt   # Prompt sent for step 1
│       └── ...
└── evaluation_logs/
    └── {with|without}_background/
        ├── {step_id}.log     # pass/fail/timeout
        └── ...
```

## Troubleshooting

### Task Not Found

Ensure your task is compiled:

```bash
task compile tasks/your_task/
```

### Test Failures

Run with gold mode to verify task setup:

```bash
inspect eval scicode.py -T mode=gold -T problems=your_task --model mockllm/model
```

### API Errors

Check your API key is set and valid:

```bash
echo $GOOGLE_API_KEY
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $TOGETHER_API_KEY
```
