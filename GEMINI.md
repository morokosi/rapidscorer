# RapidScorer Experiment Reproduction

This project reproduces the RapidScorer experiment (KDD 2018) and compares it with QuickScorer (QS) and Vectorized-QuickScorer (VQS) using LightGBM ranking models on the MSLR-WEB10K dataset.

## Project Overview

The project evaluates the inference performance of GBDT models using various optimized algorithms:
- **LightGBM C API**: Standard baseline.
- **QuickScorer (QS)**: Bit-parallel tree traversal.
- **Vectorized-QuickScorer (VQS)**: SIMD-optimized version of QuickScorer.
- **RapidScorer (RS)**: Advanced optimization using Epitome structures, EqNode merging, and ByteTransposition layout.

### Technologies
- **Python**: Used for model training (`lightgbm`, `scikit-learn`) and orchestration.
- **C**: Used for high-performance inference implementations.
- **SIMD**: AVX2 and AVX512 (where supported) for data-level parallelism.
- **uv**: Python package and environment management.

## Directory Structure

```
/home/morokosi/work/rapidscorer/
├── bin/                # Compiled C binaries (benchmark, test_correctness)
├── include/            # LightGBM C API headers
├── lib_lightgbm.so     # LightGBM shared library
├── models/             # Trained LightGBM models in .bin and .txt formats
├── MSLR-WEB10K/        # Dataset directory
├── src/
│   ├── c/              # C inference engine and benchmark source code
│   │   ├── benchmark.c         # Main benchmarking harness
│   │   ├── test_correctness.c  # Correctness verification against LightGBM
│   │   ├── qs_conversion.h     # Logic to convert raw trees to QS/RS formats
│   │   ├── vqs_rs_impl.h       # VQS and RS SIMD implementation details
│   │   └── common.h, model.h, qs.h, model_util.h # Data structures and utils
│   └── python/         # Python scripts for training and orchestration
│       ├── train_grid.py       # Trains models across tree/leaf ranges
│       └── run_grid.py         # Main orchestration script
├── Makefile            # Build management for C components
├── REPORT.md           # Summary of experimental findings
├── kdd_rapidscorer.pdf # RapidScorer paper
├── paper.pdf           # QuickScorer / VectorizedQuickScorer paper
└── results.tsv         # Detailed benchmark output
```

## Building and Running

### Prerequisites
- GCC with AVX2/AVX512 support.
- `uv` for Python environment management.
- LightGBM shared library (`lib_lightgbm.so`) and headers in `include/`.

### Automated Execution
The entire pipeline (training, building, benchmarking, and testing) can be run with a single command from the project root:
```bash
python3 src/python/run_grid.py
```

### Manual Build (C Components)
To compile the inference engines and benchmarking tools:
```bash
make
```
Binaries will be placed in the `bin/` directory.

### Running Benchmarks
To run the benchmark for a specific model:
```bash
bin/benchmark <model_bin> <model_txt> <test_txt>
├── bin/                # Compiled C binaries (benchmark, test_correctness, unit_tests)
...
### Running Correctness Tests
To verify that the custom implementations match the LightGBM baseline:
```bash
bin/test_correctness <model_bin> <model_txt> <test_txt>
```

### Unit Testing
Isolated unit tests for core inference logic using simple hand-crafted trees can be run via:
```bash
bin/unit_tests
```

## Development Conventions

- **Performance First**: Core inference loops are in C using SIMD intrinsics.
- **Data Layout**: For RapidScorer, adheres to the memory layouts described in the RapidScorer paper (Epitome, EqNode, ByteTransposition).
- **Verification**: Every model must pass the correctness test against the standard LightGBM `LGBM_BoosterPredictForMatSingleRow` output.
- **Unit Testing**: All `score` functions (QS, VQS, RS) must have associated unit tests in `src/c/unit_tests.c` to verify logic with simple, deterministic models.
- **Modular Conversion**: Conversion from raw trees to optimized structures is isolated in `qs_conversion.h` to ensure consistency between QS, VQS, and RS.

