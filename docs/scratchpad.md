./scripts/eval_trm_checkpoint.sh --checkpoint checkpoints/etrm-final/F1_standard/step_174622 --max-eval-groups 32 --config-overrides global_batch_size=512

./scripts/eval_trm_checkpoint.sh --checkpoint checkpoints/etrm-final/F3_etrmtrm/step_87310 --max-eval-groups 32 --config-overrides global_batch_size=1024

./scripts/eval_trm_checkpoint.sh --checkpoint checkpoints/etrm-final/F2_hybrid_var/step_174240 --max-eval-groups 32 --config-overrides global_batch_size=1024
