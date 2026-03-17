#ifndef VQS_RS_IMPL_H
#define VQS_RS_IMPL_H

#include <immintrin.h>
#include <string.h>
#include "qs.h"

#define V_RS 32
#define V_RS_512 64
#define V_QS 4

static inline void vqs_score_avx2(QSModel *model, double **features_v, bitvector **v_v, double *scores_v) {
    for (int s = 0; s < V_QS; s++) {
        for (int i = 0; i < model->num_trees; i++) v_v[s][i] = model->init_masks[i];
    }

    for (int k = 0; k < model->num_features; k++) {
        int idx = model->feature_offsets[k]; if (idx == -1) continue;
        int next_feat_idx = model->num_thresholds;
        for (int next_k = k + 1; next_k < model->num_features; next_k++) {
            if (model->feature_offsets[next_k] != -1) { next_feat_idx = model->feature_offsets[next_k]; break; }
        }

        __m256d feat_v = _mm256_set_pd(features_v[3][k], features_v[2][k], features_v[1][k], features_v[0][k]);
        while (idx < next_feat_idx) {
            QSThreshold *t = &model->thresholds[idx];
            __m256d thresh_v = _mm256_set1_pd(t->val);
            __m256d cmp_m = _mm256_cmp_pd(feat_v, thresh_v, _CMP_GT_OQ);
            int movemask = _mm256_movemask_pd(cmp_m);

            if (movemask == 0) break;
            for (int s = 0; s < V_QS; s++) if (movemask & (1 << s)) v_v[s][t->tree_idx] &= t->mask;
            idx++;
        }
    }

    for (int s = 0; s < V_QS; s++) {
        scores_v[s] = 0;
        for (int i = 0; i < model->num_trees; i++) scores_v[s] += model->leaf_values[i][__builtin_ctzll(v_v[s][i])];
    }
}

static inline __m256i expand_bits_to_bytes_avx2(uint32_t m) {
    __m256i v_m = _mm256_set1_epi32(m);
    __m256i shuf_mask = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
    );
    __m256i bytes = _mm256_shuffle_epi8(v_m, shuf_mask);
    __m256i pow2_v = _mm256_setr_epi8(
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7,
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7,
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7,
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7
    );
    __m256i res = _mm256_and_si256(bytes, pow2_v);
    return _mm256_cmpeq_epi8(res, pow2_v);
}

static inline void rapid_scorer_avx2(RSModel *model, double **features_v, uint8_t *leaf_indexes, double *scores_v) {
    int m_bytes = (model->max_leaves + 7) / 8;
    memset(leaf_indexes, 0xFF, model->num_trees * m_bytes * V_RS);
    
    for (int k = 0; k < model->num_features; k++) {
        int eq_idx = model->feature_offsets[k]; if (eq_idx == -1) continue;
        int next_eq_idx = -1;
        for (int next_k = k + 1; next_k <= model->num_features; next_k++) {
            if (model->feature_offsets[next_k] != -1) { next_eq_idx = model->feature_offsets[next_k]; break; }
        }

        __m256d f_v[8];
        for (int i = 0; i < 8; i++) f_v[i] = _mm256_set_pd(features_v[i*4+3][k], features_v[i*4+2][k], features_v[i*4+1][k], features_v[i*4+0][k]);

        while (eq_idx < next_eq_idx) {
            EqNode *eq = &model->eqnodes[eq_idx];
            uint32_t eta = 0;
            __m256d t_v = _mm256_set1_pd(eq->theta);
            for (int i = 0; i < 8; i++) eta |= ((uint32_t)_mm256_movemask_pd(_mm256_cmp_pd(f_v[i], t_v, _CMP_LE_OQ)) << (i * 4));
            
            if (eta == 0xFFFFFFFF) break;

            __m256i m_mid = expand_bits_to_bytes_avx2(eta);

            for (int q = 0; q < eq->u; q++) {
                int t_id = eq->tree_ids[q]; Epitome *ep = &eq->epitomes[q];
                int base = (t_id * m_bytes) * V_RS;
                
                __m256i target_fbp = _mm256_loadu_si256((__m256i*)&leaf_indexes[base + ep->fbp * V_RS]);
                _mm256_storeu_si256((__m256i*)&leaf_indexes[base + ep->fbp * V_RS], _mm256_and_si256(target_fbp, _mm256_or_si256(m_mid, _mm256_set1_epi8(ep->fb))));

                if (ep->fbp != ep->ebp) {
                    for (int b = ep->fbp + 1; b < ep->ebp; b++) {
                        __m256i target_b = _mm256_loadu_si256((__m256i*)&leaf_indexes[base + b * V_RS]);
                        _mm256_storeu_si256((__m256i*)&leaf_indexes[base + b * V_RS], _mm256_and_si256(target_b, m_mid));
                    }
                    __m256i target_ebp = _mm256_loadu_si256((__m256i*)&leaf_indexes[base + ep->ebp * V_RS]);
                    _mm256_storeu_si256((__m256i*)&leaf_indexes[base + ep->ebp * V_RS], _mm256_and_si256(target_ebp, _mm256_or_si256(m_mid, _mm256_set1_epi8(ep->eb))));
                }
            }
            eq_idx++;
        }
    }
    
    memset(scores_v, 0, sizeof(double) * V_RS);
    for (int t = 0; t < model->num_trees; t++) {
        uint8_t *base = &leaf_indexes[t * m_bytes * V_RS];
        uint8_t b_arr[V_RS]; int i_arr[V_RS];
        memset(b_arr, 0, V_RS); memset(i_arr, 0, sizeof(int) * V_RS);
        uint32_t fin = 0;
        for (int b = 0; b < m_bytes; b++) {
            uint32_t mask = ~_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i*)&base[b * V_RS]), _mm256_setzero_si256()));
            uint32_t newly = mask & ~fin;
            while (newly > 0) { int s = __builtin_ctz(newly); b_arr[s] = base[b * V_RS + s]; i_arr[s] = b; newly &= ~(1U << s); }
            fin |= mask; if (fin == 0xFFFFFFFFU) break;
        }
        for (int s = 0; s < V_RS; s++) if (b_arr[s] != 0) scores_v[s] += model->leaf_values[t][i_arr[s] * 8 + __builtin_ctz(b_arr[s])];
    }
}

#ifdef __AVX512F__
static inline void rapid_scorer_avx512(RSModel *model, double **features_v, uint8_t *leaf_indexes, double *scores_v) {
    int m_bytes = (model->max_leaves + 7) / 8;
    memset(leaf_indexes, 0xFF, model->num_trees * m_bytes * V_RS_512);
    for (int k = 0; k < model->num_features; k++) {
        int eq_idx = model->feature_offsets[k]; if (eq_idx == -1) continue;
        int next_eq_idx = -1;
        for (int next_k = k + 1; next_k <= model->num_features; next_k++) {
            if (model->feature_offsets[next_k] != -1) { next_eq_idx = model->feature_offsets[next_k]; break; }
        }
        __m512d f_v[8];
        for (int i = 0; i < 8; i++) f_v[i] = _mm512_set_pd(features_v[i*8+7][k], features_v[i*8+6][k], features_v[i*8+5][k], features_v[i*8+4][k], features_v[i*8+3][k], features_v[i*8+2][k], features_v[i*8+1][k], features_v[i*8+0][k]);
        
        while (eq_idx < next_eq_idx) {
            EqNode *eq = &model->eqnodes[eq_idx];
            uint64_t eta = 0;
            __m512d t_v = _mm512_set1_pd(eq->theta);
            for (int i = 0; i < 8; i++) eta |= ((uint64_t)_mm512_cmp_pd_mask(f_v[i], t_v, _CMP_LE_OQ) << (i * 8));
            
            if (eta == 0xFFFFFFFFFFFFFFFFULL) break;
            __mmask64 k_mid = eta, k_inv = ~eta;
            __m512i ff_v = _mm512_set1_epi8(0xFF);
            for (int q = 0; q < eq->u; q++) {
                int t_id = eq->tree_ids[q]; Epitome *ep = &eq->epitomes[q];
                int base = (t_id * m_bytes) * V_RS_512;
                __m512i target_fbp = _mm512_loadu_si512((__m512i*)&leaf_indexes[base + ep->fbp * V_RS_512]);
                _mm512_storeu_si512((__m512i*)&leaf_indexes[base + ep->fbp * V_RS_512], _mm512_and_si512(target_fbp, _mm512_mask_set1_epi8(ff_v, k_inv, ep->fb)));
                if (ep->fbp != ep->ebp) {
                    for (int b = ep->fbp + 1; b < ep->ebp; b++) {
                        __m512i target_b = _mm512_loadu_si512((__m512i*)&leaf_indexes[base + b * V_RS_512]);
                        _mm512_storeu_si512((__m512i*)&leaf_indexes[base + b * V_RS_512], _mm512_maskz_mov_epi8(k_mid, target_b));
                    }
                    __m512i target_ebp = _mm512_loadu_si512((__m512i*)&leaf_indexes[base + ep->ebp * V_RS_512]);
                    _mm512_storeu_si512((__m512i*)&leaf_indexes[base + ep->ebp * V_RS_512], _mm512_and_si512(target_ebp, _mm512_mask_set1_epi8(ff_v, k_inv, ep->eb)));
                }
            }
            eq_idx++;
        }
    }
    memset(scores_v, 0, sizeof(double) * V_RS_512);
    for (int t = 0; t < model->num_trees; t++) {
        uint8_t *base = &leaf_indexes[t * m_bytes * V_RS_512];
        uint8_t b_arr[V_RS_512]; int i_arr[V_RS_512];
        memset(b_arr, 0, V_RS_512); memset(i_arr, 0, sizeof(int) * V_RS_512);
        uint64_t fin = 0;
        for (int b = 0; b < m_bytes; b++) {
            uint64_t mask = _mm512_cmpneq_epi8_mask(_mm512_loadu_si512((__m512i*)&base[b * V_RS_512]), _mm512_setzero_si512());
            uint64_t newly = mask & ~fin;
            while (newly > 0) { int s = __builtin_ctzll(newly); b_arr[s] = base[b * V_RS_512 + s]; i_arr[s] = b; newly &= ~(1ULL << s); }
            fin |= mask; if (fin == 0xFFFFFFFFFFFFFFFFULL) break;
        }
        for (int s = 0; s < V_RS_512; s++) if (b_arr[s] != 0) scores_v[s] += model->leaf_values[t][i_arr[s] * 8 + __builtin_ctz(b_arr[s])];
    }
}
#endif

#endif
