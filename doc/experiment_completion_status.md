# Experiment Completion Status

Last updated: 2026-01-03

## Summary

This document tracks the completion status of generalization ablation experiments.

## OURS Method

| Experiment | Episodes | Status |
|------------|----------|--------|
| ours_rs_mem_r | 76/76 | Complete |
| ours_rs_mem_s | 76/76 | Complete |
| ours_rs_mem_rs | 76/76 | Complete |
| ours_hrt_mem_r | 17/17 | Complete |
| ours_hrt_mem_t | **10/17** | **INCOMPLETE** |
| ours_hrt_mem_rht | 17/17 | Complete |
| ours_hrst_mem_r | - | Missing log |
| ours_hrst_mem_h | 5/5 | Complete |
| ours_hrst_mem_s | 5/5 | Complete |
| ours_hrst_mem_t | 5/5 | Complete |
| ours_hrst_mem_rhst | 5/5 | Complete |

## MEMENTO Method

| Experiment | Episodes | Status |
|------------|----------|--------|
| memento_rs_mem_r | **18/76** | **INCOMPLETE - OOM caused 9/12 workers to fail** |
| memento_rs_mem_s | **18/76** | **INCOMPLETE - OOM caused 9/12 workers to fail** |
| memento_rs_mem_rs | **18/76** | **INCOMPLETE - OOM caused 9/12 workers to fail** |
| memento_rht_mem_r | **5/17** | **INCOMPLETE** |
| memento_rht_mem_h | **16/17** | 1 episode missing |
| memento_rht_mem_t | 17/17 | Complete |
| memento_rht_mem_rht | **11/17** | **INCOMPLETE** |
| memento_rhst_mem_r | 5/5 | Complete |
| memento_rhst_mem_h | 5/5 | Complete |
| memento_rhst_mem_s | 5/5 | Complete |
| memento_rhst_mem_t | 5/5 | Complete |
| memento_rhst_mem_rhst | 5/5 | Complete |

## Issues

### Critical: Incomplete Runs
The following experiments did not complete all episodes and need to be rerun:

1. **ours_hrt_mem_t** - Only 10/17 episodes completed
2. **memento_rht_mem_r** - Only 5/17 episodes completed
3. **memento_rht_mem_rht** - Only 11/17 episodes completed
4. **memento_rs_mem_r** - Only 18/76 episodes (CUDA OOM with num_proc=12, fixed to num_proc=4)
5. **memento_rs_mem_s** - Only 18/76 episodes (CUDA OOM with num_proc=12, fixed to num_proc=4)
6. **memento_rs_mem_rs** - Only 18/76 episodes (CUDA OOM with num_proc=12, fixed to num_proc=4)

### Note: Episode Counts by Dataset
- OURS R+S: 76 episodes (19 per worker × 4 workers)
- OURS R+H+T: 17 episodes
- OURS R+H+S+T: 5 episodes
- MEMENTO R+S: 76 episodes (needs rerun with fixed num_proc=4)
- MEMENTO R+H+T: 17 episodes (partial)
- MEMENTO R+H+S+T: 5 episodes (complete)

## Jobs to Resubmit

```bash
# Rerun incomplete OURS experiments
sbatch all_scripts/slurm_files/run_ours_hrt_mem_t.slurm

# Rerun incomplete MEMENTO R+H+T experiments
sbatch all_scripts/slurm_files/run_memento_rht_mem_r.slurm
sbatch all_scripts/slurm_files/run_memento_rht_mem_rht.slurm

# Rerun MEMENTO R+S experiments (fixed num_proc=4 to avoid OOM)
sbatch all_scripts/slurm_files/run_memento_rs_mem_r.slurm
sbatch all_scripts/slurm_files/run_memento_rs_mem_s.slurm
sbatch all_scripts/slurm_files/run_memento_rs_mem_rs.slurm
```

## Model Versions

- **OURS**: Uses newer/better LLM version
- **MEMENTO**: Uses older/worse LLM version

This difference should be noted when comparing results.
