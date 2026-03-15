#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <LightGBM/c_api.h>
#include "common.h"
#include "model_util.h"
#include "qs.h"
#include "qs_conversion.h"
#include "vqs_rs_impl.h"

#define TOLERANCE 0.0

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model_bin> <model_txt> <test_txt>\n", argv[0]);
        return 1;
    }
    const char *m_bin = argv[1], *m_txt = argv[2], *t_txt = argv[3];
    Model *raw = load_model(m_bin);
    if (!raw) { fprintf(stderr, "Failed to load raw model\n"); return 1; }
    fprintf(stderr, "Loaded model (T=%d, F=%d)\n", raw->num_trees, raw->num_features);
    
    fprintf(stderr, "Converting to QS...\n");
    QSModel *qs = convert_to_qs(raw);
    fprintf(stderr, "Converting to RS...\n");
    RSModel *rs = convert_to_rs(raw);
    
    fprintf(stderr, "Loading dataset: %s\n", t_txt);
    Dataset *ds = load_dataset(t_txt, 1000, raw->num_features); 
    if (!ds) { fprintf(stderr, "Failed to load dataset\n"); return 1; }
    fprintf(stderr, "Loaded dataset (%d docs)\n", ds->num_docs);

    BoosterHandle booster; int n_iters; 
    fprintf(stderr, "Loading LightGBM text model: %s\n", m_txt);
    if (LGBM_BoosterCreateFromModelfile(m_txt, &n_iters, &booster) != 0) {
        fprintf(stderr, "LGBM Load Error: %s\n", LGBM_GetLastError());
        return 1;
    }
    fprintf(stderr, "Loaded text model\n");
    
    int n_docs = ds->num_docs;
    double *lgbm_scores = malloc(sizeof(double) * n_docs);
    double *qs_scores = malloc(sizeof(double) * n_docs);
    double *vqs_scores = malloc(sizeof(double) * n_docs);
    double *rs_scores = malloc(sizeof(double) * n_docs);

    int64_t out_len;
    for (int i = 0; i < n_docs; i++) {
        LGBM_BoosterPredictForMatSingleRow(booster, ds->docs[i].features, C_API_DTYPE_FLOAT64, raw->num_features, 1, C_API_PREDICT_RAW_SCORE, 0, -1, "", &out_len, &lgbm_scores[i]);
    }

    bitvector *v_q = malloc(sizeof(bitvector) * qs->num_trees);
    for (int i = 0; i < n_docs; i++) {
        qs_scores[i] = qs_score(qs, ds->docs[i].features, v_q);
    }

    bitvector *v_vqs_flat = malloc(sizeof(bitvector) * V_QS * qs->num_trees);
    bitvector **v_vqs = malloc(sizeof(bitvector*) * V_QS);
    for(int i=0; i<V_QS; i++) v_vqs[i] = &v_vqs_flat[i * qs->num_trees];
    double s_vqs[V_QS], **f_vqs = malloc(sizeof(double*) * V_QS);
    for (int b = 0; b < n_docs / V_QS; b++) {
        for (int i = 0; i < V_QS; i++) f_vqs[i] = ds->docs[b * V_QS + i].features;
        vqs_score_avx2(qs, f_vqs, v_vqs, s_vqs);
        for (int i = 0; i < V_QS; i++) vqs_scores[b * V_QS + i] = s_vqs[i];
    }

    int m_b = (rs->max_leaves + 7) / 8;
    uint8_t *l_i;
    if (posix_memalign((void**)&l_i, 32, rs->num_trees * m_b * V_RS) != 0) {
        fprintf(stderr, "Failed to allocate aligned memory for leaf_indexes\n");
        return 1;
    }
    double s_v_r[V_RS], **f_v_r = malloc(sizeof(double*) * V_RS);
    for (int b = 0; b < n_docs / V_RS; b++) {
        for (int i = 0; i < V_RS; i++) f_v_r[i] = ds->docs[b * V_RS + i].features;
        rapid_scorer_avx2(rs, f_v_r, l_i, s_v_r);
        for (int i = 0; i < V_RS; i++) rs_scores[b * V_RS + i] = s_v_r[i];
    }

#ifdef __AVX512F__
    double *rs512_scores = malloc(sizeof(double) * n_docs);
    uint8_t *l_i_512;
    posix_memalign((void**)&l_i_512, 64, rs->num_trees * m_b * V_RS_512);
    double s_v_r_512[V_RS_512], **f_v_r_512 = malloc(sizeof(double*) * V_RS_512);
    for (int b = 0; b < n_docs / V_RS_512; b++) {
        for (int i = 0; i < V_RS_512; i++) f_v_r_512[i] = ds->docs[b * V_RS_512 + i].features;
        rapid_scorer_avx512(rs, f_v_r_512, l_i_512, s_v_r_512);
        for (int i = 0; i < V_RS_512; i++) rs512_scores[b * V_RS_512 + i] = s_v_r_512[i];
    }
#endif

    int failed = 0;
    int check_n = (n_docs / V_RS_512) * V_RS_512; // Most restrictive batch size
    for (int i = 0; i < check_n; i++) {
        if (fabs(lgbm_scores[i] - qs_scores[i]) > TOLERANCE) {
            printf("FAIL: QS at doc %d (LGBM: %.15f, QS: %.10f, Diff: %.10e)\n", i, lgbm_scores[i], qs_scores[i], lgbm_scores[i] - qs_scores[i]);
            failed = 1;
        }
        if (fabs(lgbm_scores[i] - vqs_scores[i]) > TOLERANCE) {
            printf("FAIL: VQS at doc %d (LGBM: %.15f, VQS: %.10f, Diff: %.10e)\n", i, lgbm_scores[i], vqs_scores[i], lgbm_scores[i] - vqs_scores[i]);
            failed = 1;
        }
        if (fabs(lgbm_scores[i] - rs_scores[i]) > TOLERANCE) {
            printf("FAIL: RS_AVX2 at doc %d (LGBM: %.15f, RS: %.10f, Diff: %.10e)\n", i, lgbm_scores[i], rs_scores[i], lgbm_scores[i] - rs_scores[i]);
            failed = 1;
        }
#ifdef __AVX512F__
        if (fabs(lgbm_scores[i] - rs512_scores[i]) > TOLERANCE) {
            printf("FAIL: RS_AVX512 at doc %d (LGBM: %.15f, RS: %.10f, Diff: %.10e)\n", i, lgbm_scores[i], rs512_scores[i], lgbm_scores[i] - rs512_scores[i]);
            failed = 1;
        }
#endif
        if (failed) break;
    }

    if (!failed) printf("PASS: All implementations match LightGBM baseline.\n");

    LGBM_BoosterFree(booster); free_qs_model(qs); free_rs_model(rs); free_model(raw); free_dataset(ds);
    free(lgbm_scores); free(qs_scores); free(vqs_scores); free(rs_scores);
#ifdef __AVX512F__
    free(rs512_scores); free(l_i_512); free(f_v_r_512);
#endif
    free(v_q); free(v_vqs_flat); free(v_vqs); free(f_vqs); free(l_i); free(f_v_r);
    return failed;
}
