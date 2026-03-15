#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <LightGBM/c_api.h>
#include "common.h"

int main() {
    const char *model_file = "model.txt";
    const char *test_file = "test_subset.txt";
    int max_rows = 20000;

    BoosterHandle booster;
    int num_iters;
    if (LGBM_BoosterCreateFromModelfile(model_file, &num_iters, &booster) != 0) {
        fprintf(stderr, "Error loading model: %s\n", LGBM_GetLastError());
        return 1;
    }

    Dataset *ds = load_dataset(test_file, max_rows);
    if (!ds) {
        fprintf(stderr, "Error loading dataset\n");
        return 1;
    }

    double *out_results = malloc(sizeof(double) * ds->num_docs);
    int64_t out_len;
    
    double start_time = get_time();
    for (int i = 0; i < ds->num_docs; i++) {
        LGBM_BoosterPredictForMatSingleRow(
            booster,
            ds->docs[i].features,
            C_API_DTYPE_FLOAT32,
            ds->num_features,
            1, // is_row_major
            C_API_PREDICT_RAW_SCORE,
            0,
            -1,
            "",
            &out_len,
            &out_results[i]
        );
    }
    double end_time = get_time();

    printf("LightGBM Baseline:\n");
    printf("Total documents: %d\n", ds->num_docs);
    printf("Total time: %.6f s\n", end_time - start_time);
    printf("Time per document: %.6f ms\n", (end_time - start_time) * 1000.0 / ds->num_docs);

    // Save predictions for verification
    FILE *fp = fopen("lgb_baseline_predictions.txt", "w");
    for (int i = 0; i < ds->num_docs; i++) {
        fprintf(fp, "%.10f\n", out_results[i]);
    }
    fclose(fp);

    free(out_results);
    free_dataset(ds);
    LGBM_BoosterFree(booster);

    return 0;
}
