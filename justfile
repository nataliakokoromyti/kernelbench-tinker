# KernelBench RL Training Commands

# Default run name (can be overridden: just train run_name=my_experiment)
run_name := "run_" + `date +%Y%m%d_%H%M%S`
config := "src/kernelbench_tinker/config/rl_kernelbench.yaml"
runs_dir := "./runs"

# List available commands
default:
    @just --list

# === Training ===

# Start training (detached, survives shell close)
train run=run_name:
    @mkdir -p {{runs_dir}}
    @echo "Starting training: {{run}}"
    nohup uv run python -m kernelbench_tinker.scripts.train_kernel_rl \
        --config {{config}} \
        log_path={{runs_dir}}/{{run}} \
        > {{runs_dir}}/{{run}}_nohup.log 2>&1 &
    @sleep 2
    @pgrep -f "log_path={{runs_dir}}/{{run}}" > /dev/null && echo "✓ Training started (PID: $$(pgrep -f 'log_path={{runs_dir}}/{{run}}'))" || echo "✗ Failed to start"
    @echo "Logs: {{runs_dir}}/{{run}}/logs.log"
    @echo "Metrics: {{runs_dir}}/{{run}}/metrics.jsonl"

# Start training with custom config
train-config config_path run=run_name:
    @mkdir -p {{runs_dir}}
    @echo "Starting training: {{run}} with config: {{config_path}}"
    nohup uv run python -m kernelbench_tinker.scripts.train_kernel_rl \
        --config {{config_path}} \
        log_path={{runs_dir}}/{{run}} \
        > {{runs_dir}}/{{run}}_nohup.log 2>&1 &
    @sleep 2
    @pgrep -f "log_path={{runs_dir}}/{{run}}" > /dev/null && echo "✓ Training started" || echo "✗ Failed to start"

# Resume training from checkpoint
resume run:
    @echo "Resuming training: {{run}}"
    nohup uv run python -m kernelbench_tinker.scripts.train_kernel_rl \
        --config {{config}} \
        log_path={{runs_dir}}/{{run}} \
        load_checkpoint_path={{runs_dir}}/{{run}} \
        > {{runs_dir}}/{{run}}_nohup.log 2>&1 &
    @sleep 2
    @pgrep -f "log_path={{runs_dir}}/{{run}}" > /dev/null && echo "✓ Training resumed" || echo "✗ Failed to start"

# === Monitoring ===

# Show live logs for a run
logs run:
    @tail -f {{runs_dir}}/{{run}}/logs.log

# Show last N lines of logs
logs-tail run n="50":
    @tail -n {{n}} {{runs_dir}}/{{run}}/logs.log

# Show batch metrics for a run
metrics run:
    @echo "=== Metrics for {{run}} ==="
    @wc -l < {{runs_dir}}/{{run}}/metrics.jsonl | xargs -I {} echo "Batches completed: {}"
    @echo "---"
    @cat {{runs_dir}}/{{run}}/metrics.jsonl | uv run python3 -c "import sys,json; [print(f\"Batch {d['step']}: reward={d['reward/mean']:.3f} (±{d['reward/std']:.3f}), compile={d['kernel/compile_rate']*100:.1f}%, correct={d['kernel/correct_rate']*100:.1f}%\") for d in (json.loads(l) for l in sys.stdin)]"

# Watch metrics update live
watch-metrics run:
    watch -n 10 'echo "=== {{run}} ===" && \
        wc -l < {{runs_dir}}/{{run}}/metrics.jsonl | xargs -I {} echo "Batches: {}" && \
        tail -1 {{runs_dir}}/{{run}}/metrics.jsonl 2>/dev/null | uv run python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"Latest: reward={d[\"reward/mean\"]:.3f}, compile={d[\"kernel/compile_rate\"]*100:.1f}%, correct={d[\"kernel/correct_rate\"]*100:.1f}%\")" 2>/dev/null || echo "No metrics yet"'

# Show summary of all runs
summary:
    @echo "=== All Runs ==="
    @for dir in {{runs_dir}}/*/; do \
        name=$$(basename $$dir); \
        if [ -f "$$dir/metrics.jsonl" ]; then \
            batches=$$(wc -l < "$$dir/metrics.jsonl"); \
            echo "$$name: $$batches batches"; \
        fi; \
    done

# === Process Management ===

# Check if training is running
status:
    @echo "=== Running Training Jobs ==="
    @pgrep -fa "train_kernel_rl" | grep -v grep || echo "No training jobs running"

# Stop a specific run
stop run:
    @echo "Stopping {{run}}..."
    @pkill -f "log_path={{runs_dir}}/{{run}}" && echo "✓ Stopped" || echo "Not running"

# Stop all training jobs
stop-all:
    @echo "Stopping all training jobs..."
    @pkill -f "train_kernel_rl" && echo "✓ All stopped" || echo "No jobs running"

# === TensorBoard ===

# Launch TensorBoard for a run
tensorboard run port="6006":
    uv run tensorboard --logdir {{runs_dir}}/{{run}}/tensorboard --port {{port}}

# Launch TensorBoard for all runs
tensorboard-all port="6006":
    uv run tensorboard --logdir {{runs_dir}} --port {{port}}

# === Utilities ===

# List all runs
list:
    @ls -lt {{runs_dir}} | head -20

# Show disk usage of runs
disk:
    @du -sh {{runs_dir}}/*/ 2>/dev/null | sort -h

# Clean up a run (DANGEROUS)
clean run:
    @echo "This will delete {{runs_dir}}/{{run}}"
    @read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf {{runs_dir}}/{{run}} && echo "✓ Deleted" || echo "Cancelled"

# Show uniform reward warnings count
uniform-warnings run:
    @echo "Uniform reward warnings in {{run}}:"
    @grep -c "All rewards are uniform" {{runs_dir}}/{{run}}/logs.log 2>/dev/null || echo "0"

