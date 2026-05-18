#!/usr/bin/env bash
set -uo pipefail

# MODEL="allenai/OLMo-2-0425-1B"
# MODEL="allenai/OLMo-2-1124-7B"
# MODEL="allenai/Olmo-3-1025-7B"
# MODEL="allenai/OLMoE-1B-7B-0924"
# MODEL="allenai/Olmo-Hybrid-7B"
# MODEL="allenai/OLMo-2-1124-13B"
MODEL="allenai/Olmo-3-1125-32B"
# MODEL="allenai/OLMo-2-0325-32B"

OUTPUT_DIR="/weka/oe-training-default/yashasbls/georges-functional-analysis"
TARGET_TOKENS=6000000
NUM_GPUS=8
# global_batch_size controls which instances are dropped (always 16 seqs = 65536 tokens).
# compute_batch_size controls GPU memory — reduce for 7B/32B if you OOM.
GLOBAL_BATCH_SIZE=65536   # 16 * 4096: fixed for consistent truncation across all models
COMPUTE_BATCH_SIZE=8      # sequences per forward pass (reduce for larger models)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Fetch all stage1 revisions from HuggingFace in fractal order:
# endpoints first, then midpoints, then quarter points, etc.
# This way you get a coarse trajectory early and refine over time.
REVISIONS=$(curl -sL "https://huggingface.co/api/models/${MODEL}/refs" | \
    python3 -c "
import sys, json

data = json.load(sys.stdin)
branches = [b['name'] for b in data['branches'] if b['name'].startswith('stage1')]
branches.sort(key=lambda x: int(x.split('-step')[1].split('-')[0]))

# Fractal (binary subdivision) ordering
def fractal_order(items):
    if len(items) <= 2:
        return list(items)
    result = [items[0], items[-1]]
    queue = [(0, len(items) - 1)]
    while queue:
        next_queue = []
        for lo, hi in queue:
            if hi - lo <= 1:
                continue
            mid = (lo + hi) // 2
            result.append(items[mid])
            next_queue.append((lo, mid))
            next_queue.append((mid, hi))
        queue = next_queue
    return result

ordered = fractal_order(branches)
print(' '.join(ordered))
")

NUM_REVISIONS=$(echo "$REVISIONS" | wc -w)
echo "Found ${NUM_REVISIONS} stage1 revisions for ${MODEL}"

cd "$REPO_ROOT"

while true; do
    uv run torchrun --nproc-per-node="$NUM_GPUS" \
        src/scripts/analyze/collect_token_logprobs_hf.py \
        --work-dir /weka/oe-training-default/yashasbls/georges-functional-analysis/.dataset_cache \
        --model "$MODEL" \
        --revisions $REVISIONS \
        --output-dir "$OUTPUT_DIR" \
        --target-tokens "$TARGET_TOKENS" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --compute-batch-size "$COMPUTE_BATCH_SIZE" \
        --top-k 32 && break
    echo "Command exited with code $?. Restarting in 5 seconds..."
    sleep 5
done
