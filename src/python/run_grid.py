import os
import subprocess

def main():
    # 1. Train models (when necessary)
    # print("--- Phase 1: Training Grid of Models ---")
    # subprocess.run(["uv", "run", "--with", "lightgbm==4.3.0", "--with", "scikit-learn", "--with", "numpy<2", "python", "src/python/train_grid.py"], check=True)

    # 2. Compile benchmark
    print("\n--- Phase 2: Compiling Benchmark ---")
    subprocess.run(["make"], check=True)

    # 3. Run benchmarks
    print("\n--- Phase 3: Running Benchmarks ---")
    tree_range = [10, 100, 500, 1000]
    leaf_range = [16, 32, 64]
    
    test_txt = "test_subset.txt"
    if not os.path.exists(test_txt):
        print(f"Creating {test_txt}...")
        # Assume MSLR-WEB10K is in root
        subprocess.run(["head", "-n", "20000", "MSLR-WEB10K/Fold1/test.txt"], stdout=open(test_txt, "w"))

    results = []
    header = "Trees\tLeaves\tAlgorithm\tTime_per_doc_ms"
    results.append(header)

    for n_trees in tree_range:
        for n_leaves in leaf_range:
            model_bin = f"models/model_t{n_trees}_l{n_leaves}.bin"
            model_txt = f"models/model_t{n_trees}_l{n_leaves}.txt"
            
            print(f"Benchmarking: T={n_trees}, L={n_leaves}")
            process = subprocess.Popen(["bin/benchmark", model_bin, model_txt, test_txt], 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"Error benchmarking {model_bin}: {stderr}")
                continue
            
            for line in stdout.strip().split("\n"):
                if "\t" in line:
                    results.append(f"{n_trees}\t{n_leaves}\t{line}")

    with open("results.tsv", "w") as f:
        f.write("\n".join(results) + "\n")

    # 4. Correctness Tests
    print("\n--- Phase 4: Correctness Testing ---")
    for n_trees in tree_range:
        for n_leaves in leaf_range:
            model_bin = f"models/model_t{n_trees}_l{n_leaves}.bin"
            model_txt = f"models/model_t{n_trees}_l{n_leaves}.txt"
            print(f"Testing Correctness: T={n_trees}, L={n_leaves}")
            ret = subprocess.run(["bin/test_correctness", model_bin, model_txt, test_txt])
            if ret.returncode != 0:
                print(f"!!! CORRECTNESS FAILED for T={n_trees}, L={n_leaves} !!!")
            else:
                print(f"PASS for T={n_trees}, L={n_leaves}")

    print("\nAll processes complete. Results saved to results.tsv")

if __name__ == "__main__":
    main()
