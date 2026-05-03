# Generated Results

Generated result files are intentionally excluded from the public package.
The checked-in `results/benchmark/` directory is an output root for benchmark
JSON files.

Populate it by running the benchmark scripts and reporting utilities:

- `results/benchmark/` for per-dataset benchmark JSON
- timing scripts create `results/timing/`
- unwatermarked ViSQOL scripts create `results/visqol_unwm/`

Reduced-sample validation runs should be stored separately from the main benchmark
outputs so they are not confused with full evaluation runs.
