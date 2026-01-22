This is the official codebase for the paper "Less is More: Recursive Reasoning with Tiny Networks". I'm conducting a research project to extend this work, which is described in @docs/proposal.md. I'm using this codebase as a starting point. I need to be able to reproduce the results of the paper and compare my results to the paper's results. Hence, we need to make sure that the changes we make don't alter the original behavior of the codebase.

See @docs/CODEBASE_UNDERSTANDING.md for a detailed understanding of the codebase and @docs/data_flow.md for a detailed understanding of the data flow.


# Experiments Scheduling
For running experiments, add a new line to @jobs.txt file, for instance:
```
[ ] torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py --config-name cfg_pretrain_arc_agi_1 +project_name="mmi-714" +run_name="experiment-name"
```

The scheduler will then run the command and update the line to [x] if the experiment succeeds, [!] if it fails, or [-] if it is running and append the PID of the job to the line.

Use same project-name for all experiments and assign a unique and descriptive run-name for each experiment.

# Weights & Biases
W&B entity name: `bdsaglam`

# Hydra Config
When adding new config parameters, update ALL locations:
1. **Python dataclass** (e.g., `DemoEncoderConfig` in `models/encoders/base.py`)
2. **Architecture YAML** (e.g., `config/arch/etrm.yaml` or `config/arch/etrmtrm.yaml`)

Hydra requires parameters exist in YAML for `param=value` override syntax. Missing YAML entries cause "Key not in struct" errors.

## Writing Job Commands
**Always check exact YAML parameter names before writing job commands.** Read the YAML file first.

Example: Architecture params are in `config/arch/etrm.yaml`:
- ✅ `arch.encoder_num_layers` (exists in YAML)
- ❌ `arch.num_layers` (doesn't exist, will fail)

# Adding New Config Parameters
When adding a new parameter (e.g., `lpn_latent_dim`):
1. Add to dataclass in `models/encoders/base.py` (Python)
2. Add to `config/arch/etrm.yaml` (if used by ETRM)
3. Add to `config/arch/etrmtrm.yaml` (if used by ETRMTRM)
4. Update any job commands that need the new parameter

# Adding New Encoder Types
When adding a new encoder type, update ALL these locations:
1. `models/encoders/<new_encoder>.py` - Create the encoder class
2. `models/encoders/base.py` - Add to `encoder_map` in `create_encoder()`
3. `models/encoders/__init__.py` - Add import and `__all__` export
4. **`models/recursive_reasoning/etrm.py`** - Add import AND elif case in `TRMWithEncoder.__init__`

Note: `etrm.py` has its own encoder creation logic that duplicates `create_encoder()`. Both must be updated.

