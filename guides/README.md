# Guides

This folder contains run instructions for the FlexGen GPU-server workflow.

Start here:

- [PIPELINE_ONE_COMMAND.md](PIPELINE_ONE_COMMAND.md) - one command to run FlexGen tests, policy search, and print/save the 14 policy parameters.
- [LOCAL_SYNTHETIC_FLEXGEN_TEST.md](LOCAL_SYNTHETIC_FLEXGEN_TEST.md) - local no-GPU/no-Qwen-weight test script with synthetic parameters.
- [LOCAL_GPU_SMALL_MODEL_TEST.md](LOCAL_GPU_SMALL_MODEL_TEST.md) - local RTX 4050 smoke test with a downloaded small model.
- [LOCAL_GPU_TINYLLAMA_1B_TEST.md](LOCAL_GPU_TINYLLAMA_1B_TEST.md) - local RTX 4050 test with TinyLlama 1.1B Chat.
- [FLEXGEN_PLUS_CONVEX_OPTIMIZATION_COMPARISON.md](FLEXGEN_PLUS_CONVEX_OPTIMIZATION_COMPARISON.md) - explanation of FlexGen vs a proposed convex-relaxed optimizer and how to present it.
- `config_flexgen_local_gpu.yml` - local GPU smoke-test config for `models/smollm2-135m-instruct`.
- `config_flexgen_tinyllama_local_gpu.yml` - local 1B-class GPU test config for `models/tinyllama-1.1b-chat`.
- [FLEXGEN_QWEN_GPU_SERVER_RUNBOOK.md](FLEXGEN_QWEN_GPU_SERVER_RUNBOOK.md) - full setup, GPU checks, Qwen run commands, outputs, plots, and troubleshooting.
- [FLEXGEN_QWEN_COMMANDS_ONLY.md](FLEXGEN_QWEN_COMMANDS_ONLY.md) - compact copy-paste command list for the server.
- [QWEN_ACTUAL_INFERENCE_RUNBOOK.md](QWEN_ACTUAL_INFERENCE_RUNBOOK.md) - actual Qwen model loading and text generation on GPU.
