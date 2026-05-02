# FlexGen Plus Convex Optimization Comparison

This guide explains how to present the current FlexGen optimizer and a possible second convex-relaxed optimizer in the final project.

## Main Idea

The project is optimizing **LLM serving policy**, not changing the LLM weights.

Given:

```text
model architecture
GPU memory
CPU memory
disk capacity
PCIe bandwidth
disk bandwidth
GPU compute speed
prompt length
decode length
```

The optimizer predicts a deployment policy that minimizes predicted per-token latency.

The output is the same 14 FlexGen-style parameters:

```text
1. gpu_batch_size
2. num_gpu_batches
3. compression
4. cpu_compute_delegate
5. overlap_io_compute
6. weights_gpu
7. weights_cpu
8. weights_disk
9. kv_cache_gpu
10. kv_cache_cpu
11. kv_cache_disk
12. activations_gpu
13. activations_cpu
14. activations_disk
```

The derived value:

```text
block_size = gpu_batch_size * num_gpu_batches
```

is shown separately because it is not an independent decision variable.

## What FlexGen Currently Does

The current implementation follows the FlexGen-style policy search:

```text
outer enumeration over discrete choices
+
inner linear programming over placement fractions
```

The outer search tries discrete choices:

```text
gpu_batch_size
num_gpu_batches
compression
cpu_compute_delegate
overlap_io_compute
```

The inner LP solves continuous placement:

```text
weights_gpu / weights_cpu / weights_disk
kv_cache_gpu / kv_cache_cpu / kv_cache_disk
activations_gpu / activations_cpu / activations_disk
```

So the current optimizer is already using linear programming. For every discrete policy candidate, it solves a real LP to find the best memory placement.

## Why A Pure Convex Optimizer Cannot Directly Solve All 14 Parameters

Some of the 14 parameters are discrete:

```text
gpu_batch_size: integer
num_gpu_batches: integer
compression: binary/category
cpu_compute_delegate: binary
overlap_io_compute: binary
```

Pure convex optimization is naturally suited for continuous variables, not integer or binary choices.

The 9 placement fractions are convex-friendly:

```text
weights_gpu/cpu/disk
kv_cache_gpu/cpu/disk
activations_gpu/cpu/disk
```

They obey linear constraints:

```text
weights_gpu + weights_cpu + weights_disk = 1
kv_cache_gpu + kv_cache_cpu + kv_cache_disk = 1
activations_gpu + activations_cpu + activations_disk = 1
```

and memory constraints:

```text
GPU usage <= available GPU memory
CPU usage <= available CPU memory
disk usage <= available disk capacity
```

That is why the inner problem is a linear program.

## Proposed Additional Method: Convex-Relaxed Policy Optimizer

We can add a second optimizer called:

```text
Convex Relaxed Policy Optimizer
```

The idea:

```text
1. Generate a set of candidate policies.
2. Assign each candidate a continuous selection weight between 0 and 1.
3. Solve a convex/linear program over those weights.
4. Convert the relaxed solution back into one concrete policy.
5. Output the same 14 parameters.
```

This gives another optimization strategy to compare against FlexGen.

## How The Convex Relaxation Would Work

Let each candidate policy be indexed by `i`.

Each policy has:

```text
latency_i
gpu_memory_i
cpu_memory_i
disk_memory_i
14 output parameters
```

Decision variable:

```text
x_i = fraction/weight assigned to candidate policy i
```

Constraints:

```text
sum_i x_i = 1
x_i >= 0
weighted GPU memory <= GPU capacity
weighted CPU memory <= CPU capacity
weighted disk memory <= disk capacity
```

Objective:

```text
minimize sum_i x_i * latency_i
```

This is a linear program, so it is convex.

Then we convert it back to a single policy:

```text
pick candidate with highest x_i
```

or:

```text
round the relaxed result to nearest valid policy
```

## How It Would Produce The Same 14 Parameters

The relaxed optimizer would internally work with candidate weights, but the final output would still be:

```text
gpu_batch_size
num_gpu_batches
compression
cpu_compute_delegate
overlap_io_compute
weights_gpu/cpu/disk
kv_cache_gpu/cpu/disk
activations_gpu/cpu/disk
```

So the final result looks the same as FlexGen.

The difference is the optimization method:

```text
FlexGen:
  enumerate discrete choices, solve LP for placement

Convex Relaxed:
  solve one LP over candidate-policy weights, then round/pick final policy
```

## What To Compare In The Final Demo

Show this table:

```text
Method                      Predicted latency     Predicted throughput     Notes
Manual fp16 baseline         high                  low                      naive policy
Fixed-policy LP baseline     medium/high           medium                   LP placement only
Convex-relaxed optimizer     low                   high                     relaxed global policy selection
FlexGen policy search        low/best              high/best                full enumeration + LP
```

For each method, show:

```text
per_token_latency_ms
throughput_tok_s
t_block_ms
gpu_batch_size
num_gpu_batches
compression
placement fractions
```

## How To Explain This To The Professor

Use this wording:

> We are optimizing LLM serving configuration, not training the model. The optimizer reads the model architecture and hardware measurements, then predicts the best batching, compression, and memory-placement policy. FlexGen uses enumeration plus linear programming. We also propose a convex-relaxed policy optimizer that approximates the discrete search as a convex LP over candidate policies. Both produce the same 14 deployment parameters, and we compare them against manual baselines.

## What This Shows

This demonstrates:

```text
1. The project uses actual optimization, not heuristics only.
2. The placement problem is solved with LP.
3. The system returns deployment-ready parameters.
4. The optimized policies can be compared against naive baselines.
5. The same framework can compare multiple optimization strategies.
```

## Important Honesty Point

Do not claim:

```text
"A pure convex optimizer directly solves all 14 discrete and continuous variables."
```

Instead say:

```text
"The continuous placement variables are solved exactly as an LP. The discrete policy variables are handled by FlexGen enumeration, and can also be approximated using a convex relaxation plus rounding."
```

That is technically correct.

## Suggested Final Project Story

Use this flow:

```text
1. Run tests.
2. Run baseline manual policy.
3. Run fixed-policy LP baseline.
4. Run FlexGen full policy search.
5. Optionally run convex-relaxed optimizer.
6. Print comparison table.
7. Run actual LLM inference to show the model executes on GPU.
8. Save all results as JSON for reproducibility.
```

Current pipeline already supports:

```text
tests
manual baseline
fixed-policy LP baselines
FlexGen optimized policy
actual model inference
comparison JSON output
```

Future enhancement:

```text
add convex-relaxed optimizer as another row in the comparison table
```

