 # KernelBench ↔ Tinker Integration

An end‑to‑end integration that lets [Tinker](https://tinker-docs.thinkingmachines.ai/) train models against [KernelBench](https://github.com/ScalingIntelligence/KernelBench) by generating kernels, evaluating them on Modal, and turning
those results into RL rewards.

## Overview

  This repo is a minimal **Tinker ↔ KernelBench integration** to fine-tune language models to write better GPU kernels (in the languages KB supports). It:

  - Uses **KernelBench** for prompts and kernel evaluation (compile/correctness/speed)
  - Uses **Tinker** for distributed LoRA fine-tuning with GRPO-style RL
  - Runs evaluations through Modal for isolated executions
  - Computes rewards directly from KernelBench evaluation results

## Quick Start
```bash
# 1) Deploy Modal app for isolated GPU eval
modal deploy src/kernelbench_tinker/modal/app.py

# 2) Start training
just train run=my_experiment

# 3) Tail logs
just logs run=my_experiment

# 4) Resume if crashed
just resume run=my_experiment
```

### Directory Structure

```
 /workspace/kernel_dev/
  ├── kernelbench-tinker/   # This integration repo
  ├── KernelBench/          # KernelBench benchmark
  └── tinker-cookbook/      # Tinker cookbook examples
```

## Setup

### 1. Clone repository
```bash
cd /workspace/kernel_dev
git clone https://github.com/nataliakokoromyti/kernelbench-tinker.git
```

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies

```bash
cd /workspace/kernel_dev/kernelbench-tinker
uv sync
```

### 4. Configure environment variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and set your Tinker API key (get it from https://console.tinker.thinkingmachines.ai):

```bash
TINKER_API_KEY=your-api-key-here
```

The `.env` file is automatically loaded when running scripts.

## Training

This repo wires Tinker RL to KernelBench evaluation. The training loop:
- Samples kernels from the model according to KernelBench problems
- Evaluates them with KernelBench via Modal using KB's eval harness
- Converts results into rewards and updates the model with GRPO‑style RL
- Saves checkpoints after every batch for crash recovery
  
### Using Justfile Commands

```bash
# Start training 
just train run=my_experiment

# Resume from checkpoint if crashed
just resume run=my_experiment

# View logs
just logs run=my_experiment

# Check training status
just status
```

### Manual Training

```bash
uv run python -m kernelbench_tinker.scripts.train_kernel_rl \
    --config src/kernelbench_tinker/config/rl_kernelbench.yaml \
    log_path=./runs/my_experiment
```

### Checkpoints and Resume

Checkpoints are saved to Tinker cloud after every batch. The checkpoint paths are recorded in `{log_path}/checkpoints.jsonl`.

```bash
# Resume training after a crash
just resume run=my_experiment

# Or manually:
uv run python -m kernelbench_tinker.scripts.train_kernel_rl \
    --config src/kernelbench_tinker/config/rl_kernelbench.yaml \
    log_path=./runs/my_experiment \
    load_checkpoint_path=./runs/my_experiment
```

**Note**: Kernel evaluation can sometimes crash the GPU with illegal memory access errors. The frequent checkpointing ensures minimal progress loss.

## TensorBoard Visualization

Training progress can be monitored in real-time using TensorBoard.

### Launch TensorBoard

```bash
# For a specific run
uv run tensorboard --logdir ./runs/my_experiment/tensorboard --port 6006

# For all runs
uv run tensorboard --logdir ./runs --port 6006
```

Then open http://localhost:6006 in your browser.

## Evaluation

Requires the Modal app to be deployed and Modal tokens configured.

### Evaluate a Checkpoint

```bash
uv run python -m kernelbench_tinker.scripts.eval_kernel_rl \
    checkpoint_path=./runs/my_experiment/checkpoints/final \
    level=1 \
    output_path=./runs/my_experiment/eval_results.json
```

### Evaluate Base Model

```bash
uv run python -m kernelbench_tinker.scripts.eval_kernel_rl \
    model_name=Qwen/Qwen2.5-Coder-7B-Instruct \
    level=1 \
    output_path=./baseline_results.json
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TINKER_API_KEY` | Yes | API key from https://console.tinker.thinkingmachines.ai |
| `KERNELBENCH_ROOT` | No | Optional local KernelBench override |
| `MODAL_TOKEN_ID` | Yes | Modal token ID |
| `MODAL_TOKEN_SECRET` | Yes | Modal token secret |

## Architecture

```
src/kernelbench_tinker/
  ├── env.py                          # Environment variable loading
  ├── envs/
  │   ├── kernelbench_client.py       # KernelBench Python API wrapper
  │   ├── kernelbench_env.py          # Single-turn RL environment
  ├── training/
  │   ├── models.py                   # Model/renderer configuration
  │   ├── reward.py                   # Reward shaping
  │   ├── loop.py                     # GRPO training loop
  │   ├── tensorboard_logger.py       # TensorBoard logging
  │   └── trace_logger.py             # JSONL trace logging
  ├── evaluation/
  │   └── eval_kernelbench.py         # Evaluation utilities
  ├── scripts/
  │   ├── train_kernel_rl.py          # Training CLI
  │   └── eval_kernel_rl.py           # Evaluation CLI
  ├── modal/
  │   ├── app.py                      # Modal eval app
  │   └── evaluator.py                # Modal evaluator client
  └── config/
      └── rl_kernelbench.yaml         # Default config
```

## Troubleshooting

### CUDA illegal memory access / Training crashes
Generated kernels can sometimes corrupt GPU memory. This is handled by:
- Checkpoints saved after every batch (`save_every: 1`)
- Resume capability: `just resume my_experiment`

If crashes are frequent:
```bash
# Clear GPU memory and restart
nvidia-smi --gpu-reset  # If needed
just resume my_experiment
```

### CUDA out of memory
Reduce `batch_size` or `group_size`:
```bash
batch_size=2 group_size=2
```

### Tinker API errors
1. Check your API key is set: `echo $TINKER_API_KEY`
2. Get a key from https://console.tinker.thinkingmachines.ai
3. Check Tinker service status

### Resume not working
Ensure `checkpoints.jsonl` exists in the run directory:
```bash
cat ./runs/my_experiment/checkpoints.jsonl
```
If empty or missing, training crashed before the first checkpoint was saved.

## References

- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
