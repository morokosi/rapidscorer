#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "common.h"
#include "model_util.h"
#include "qs.h"
#include "qs_conversion.h"

void vqs_score_8(QSModel *model, float **features_8, bitvector **v_8, float *scores_8) {
    int num_docs = 8;
    int num_trees = model->num_trees;

    for (int d = 0; d < num_docs; d++) {
        for (int i = 0; i < num_trees; i++) {
            v_8[d][i] = model->init_masks[i];
        }
        scores_8[d] = 0;
    }

    int i = 0;
    while (i < model->num_thresholds) {
        int f_idx = model->thresholds[i].feature_idx;
        
        __m256 feat_v = _mm256_set_ps(
            features_8[7][f_idx], features_8[6][f_idx], features_8[5][f_idx], features_8[4][f_idx],
            features_8[3][f_idx], features_8[2][f_idx], features_8[1][f_idx], features_8[0][f_idx]
        );

        while (i < model->num_thresholds && model->thresholds[i].feature_idx == f_idx) {
            Threshold *t = &model->thresholds[i];
            __m256 thresh_v = _mm256_set1_ps(t->val);
            
            __m256 mask_v = _mm256_cmp_ps(feat_v, thresh_v, _CMP_GT_OQ);
            int movemask = _mm256_movemask_ps(mask_v);

            if (movemask) {
                for (int d = 0; d < 8; d++) {
                    if (movemask & (1 << d)) {
                        v_8[d][t->tree_idx] &= t->mask;
                    }
                }
            }
            i++;
        }
    }

    for (int d = 0; d < num_docs; d++) {
        for (int i = 0; i < num_trees; i++) {
            int leaf_idx = __builtin_ctzll(v_8[d][i]);
            scores_8[d] += model->leaf_values[model->tree_offsets[i] + leaf_idx];
        }
    }
}

int main() {
    Model *raw_model = load_model("model.bin");
    if (!raw_model) return 1;
    QSModel *model = convert_to_qs(raw_model);

    Dataset *ds = load_dataset("test_subset.txt", 20000);
    if (!ds) return 1;

    int num_docs = ds->num_docs;
    int num_batches = num_docs / 8;
    float *all_scores = malloc(sizeof(float) * num_docs);

    float **features_8 = malloc(sizeof(float*) * 8);
    bitvector **v_8 = malloc(sizeof(bitvector*) * 8);
    for(int i=0; i<8; i++) v_8[i] = malloc(sizeof(bitvector) * model->num_trees);
    float scores_8[8];

    double start_time = get_time();
    for (int b = 0; b < num_batches; b++) {
        for (int i = 0; i < 8; i++) features_8[i] = ds->docs[b * 8 + i].features;
        vqs_score_8(model, features_8, v_8, scores_8);
        for (int i = 0; i < 8; i++) all_scores[b * 8 + i] = scores_8[i];
    }
    
    bitvector *v_single = malloc(sizeof(bitvector) * model->num_trees);
    for (int i = num_batches * 8; i < num_docs; i++) {
        all_scores[i] = qs_score(model, ds->docs[i].features, v_single);
    }
    double end_time = get_time();

    printf("Vectorized-QuickScorer (AVX2):\n");
    printf("Total documents: %d\n", num_docs);
    printf("Total time: %.6f s\n", end_time - start_time);
    printf("Time per document: %.6f ms\n", (end_time - start_time) * 1000.0 / num_docs);

    FILE *fp = fopen("lgb_baseline_predictions.txt", "r");
    if (fp) {
        double diff = 0;
        for (int i = 0; i < num_docs; i++) {
            double lgb_score;
            if (fscanf(fp, "%lf", &lgb_score) != 1) break;
            diff += fabs(lgb_score - all_scores[i]);
        }
        printf("Mean Absolute Difference from LGBM: %.10e\n", diff / num_docs);
        fclose(fp);
    }

    for(int i=0; i<8; i++) free(v_8[i]);
    free(v_8);
    free(features_8);
    free(v_single);
    free(all_scores);
    free_dataset(ds);
    free_qs_model(model);
    free_model(raw_model);

    return 0;
}
