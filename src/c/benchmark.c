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

int main(int argc, char **argv) {
    if (argc < 4) return 1;
    const char *m_bin = argv[1], *m_txt = argv[2], *t_txt = argv[3];
    Model *raw_model = load_model(m_bin);
    if (!raw_model) return 1;
    QSModel *qs_model = convert_to_qs(raw_model);
    RSModel *rs_model = convert_to_rs(raw_model);
    Dataset *ds = load_dataset(t_txt, 20000, raw_model->num_features);
    if (!ds) return 1;

    int n_docs = ds->num_docs;
    BoosterHandle booster; int n_iters; LGBM_BoosterCreateFromModelfile(m_txt, &n_iters, &booster);
    double *l_out = malloc(sizeof(double) * n_docs); int64_t o_len;
    double s_l = get_time();
    for (int i = 0; i < n_docs; i++) LGBM_BoosterPredictForMatSingleRow(booster, ds->docs[i].features, C_API_DTYPE_FLOAT64, raw_model->num_features, 1, C_API_PREDICT_RAW_SCORE, 0, -1, "", &o_len, &l_out[i]);
    printf("LGBM\t%.6f\n", (get_time() - s_l) * 1000.0 / n_docs);

    bitvector *v_q = malloc(sizeof(bitvector) * qs_model->num_trees);
    double s_q = get_time();
    for (int i = 0; i < n_docs; i++) qs_score(qs_model, ds->docs[i].features, v_q);
    printf("QuickScorer\t%.6f\n", (get_time() - s_q) * 1000.0 / n_docs);

    bitvector *v_vqs_flat = malloc(sizeof(bitvector) * V_QS * qs_model->num_trees);
    bitvector **v_vqs = malloc(sizeof(bitvector*) * V_QS);
    for(int i=0; i<V_QS; i++) v_vqs[i] = &v_vqs_flat[i * qs_model->num_trees];
    double s_vqs[V_QS], **f_vqs = malloc(sizeof(double*) * V_QS);
    double start_vqs = get_time();
    for (int b = 0; b < n_docs / V_QS; b++) {
        for (int i = 0; i < V_QS; i++) f_vqs[i] = ds->docs[b * V_QS + i].features;
        vqs_score_avx2(qs_model, f_vqs, v_vqs, s_vqs);
    }
    printf("V-QuickScorer_AVX2\t%.6f\n", (get_time() - start_vqs) * 1000.0 / n_docs);

    int m_b = (rs_model->max_leaves + 7) / 8;
    uint8_t *l_i;
    if (posix_memalign((void**)&l_i, 32, rs_model->num_trees * m_b * V_RS) != 0) return 1;
    double s_v_r[V_RS], **f_v_r = malloc(sizeof(double*) * V_RS);
    double start_rs = get_time();
    for (int b = 0; b < n_docs / V_RS; b++) {
        for (int i = 0; i < V_RS; i++) f_v_r[i] = ds->docs[b * V_RS + i].features;
        rapid_scorer_avx2(rs_model, f_v_r, l_i, s_v_r);
    }
    printf("RapidScorer_AVX2\t%.6f\n", (get_time() - start_rs) * 1000.0 / n_docs);

#ifdef __AVX512F__
    uint8_t *l_i_512;
    if (posix_memalign((void**)&l_i_512, 64, rs_model->num_trees * m_b * V_RS_512) != 0) return 1;
    double s_v_r_512[V_RS_512], **f_v_r_512 = malloc(sizeof(double*) * V_RS_512);
    double start_rs512 = get_time();
    for (int b = 0; b < n_docs / V_RS_512; b++) {
        for (int i = 0; i < V_RS_512; i++) f_v_r_512[i] = ds->docs[b * V_RS_512 + i].features;
        rapid_scorer_avx512(rs_model, f_v_r_512, l_i_512, s_v_r_512);
    }
    printf("RapidScorer_AVX512\t%.6f\n", (get_time() - start_rs512) * 1000.0 / n_docs);
    free(l_i_512); free(f_v_r_512);
#endif

    LGBM_BoosterFree(booster); 
    free_qs_model(qs_model); 
    free_rs_model(rs_model); 
    free_model(raw_model); 
    free_dataset(ds);
    free(l_out); free(v_q); 
    free(v_vqs_flat); free(v_vqs); free(f_vqs);
    free(l_i); free(f_v_r);
    return 0;
}
