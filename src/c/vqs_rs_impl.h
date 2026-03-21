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
    memset(scores_v, 0, sizeof(double) * V_RS);
    
    for (int k = 0; k < model->num_features; k++) {
        int eq_idx = model->feature_offsets[k];
        int next_eq_idx = model->feature_offsets[k+1];
        if (eq_idx == next_eq_idx) continue;

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
    
    /*
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
        */
       // ツリーのループ内（memset等の初期化は不要になります）
    for (int t = 0; t < model->num_trees; t++) {
        uint8_t *base = &leaf_indexes[t * m_bytes * V_RS];
        
        __m256i v_fin = _mm256_setzero_si256();
        __m256i v_first_byte_val = _mm256_setzero_si256();
        __m256i v_first_byte_idx = _mm256_setzero_si256();

        // -------------------------------------------------------------
        // Step 3.a: 左端のTRUEバイト (\vec{b}) と、そのチャンクインデックス (\vec{c_1}) を探す 
        // -------------------------------------------------------------
        for (int b = 0; b < m_bytes; b++) {
            __m256i v_curr_bytes = _mm256_loadu_si256((__m256i*)&base[b * V_RS]);
            
            // 非ゼロのバイトを 0xFF、ゼロを 0x00 のマスクに変換
            __m256i v_is_nonzero = ~_mm256_cmpeq_epi8(v_curr_bytes, _mm256_setzero_si256());
            
            // まだリーフが見つかっていない（~v_fin）かつ、今回非ゼロだったデータを抽出
            __m256i v_new_found = _mm256_andnot_si256(v_fin, v_is_nonzero);
            
            // \vec{b} と \vec{c_1} を更新
            v_first_byte_val = _mm256_blendv_epi8(v_first_byte_val, v_curr_bytes, v_new_found);
            v_first_byte_idx = _mm256_blendv_epi8(v_first_byte_idx, _mm256_set1_epi8(b), v_new_found);
            
            v_fin = _mm256_or_si256(v_fin, v_new_found);
            
            // 全てのサンプルが到達リーフを見つけたら早期退出
            if ((uint32_t)_mm256_movemask_epi8(v_fin) == 0xFFFFFFFFU) break;
        }

        // -------------------------------------------------------------
        // Step 3.b: 各バイト内の最も右にある1のビット位置 (\vec{c_2}) を計算 
        // -------------------------------------------------------------
        // AVX2の pshufb を使ったテーブル引きによる 8-bit CTZ (Count Trailing Zeros) のベクトル計算
        __m256i v_ctz_lut = _mm256_setr_epi8(
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
        );
        __m256i v_lo_nibble = _mm256_and_si256(v_first_byte_val, _mm256_set1_epi8(0x0F));
        // 8ビット右シフト命令がないため、16ビットシフトしてマスクするハック
        __m256i v_hi_nibble = _mm256_and_si256(_mm256_srli_epi16(v_first_byte_val, 4), _mm256_set1_epi8(0x0F));
        
        __m256i v_ctz_lo = _mm256_shuffle_epi8(v_ctz_lut, v_lo_nibble);
        __m256i v_ctz_hi = _mm256_shuffle_epi8(v_ctz_lut, v_hi_nibble);
        
        __m256i v_is_lo_zero = _mm256_cmpeq_epi8(v_lo_nibble, _mm256_setzero_si256());
        
        // 下位4ビットがゼロなら「上位のCTZ + 4」、そうでなければ「下位のCTZ」を \vec{c_2} とする
        __m256i v_c2 = _mm256_blendv_epi8(v_ctz_lo, _mm256_add_epi8(v_ctz_hi, _mm256_set1_epi8(4)), v_is_lo_zero);

        // -------------------------------------------------------------
        // Step 3.c: 最終インデックスの算出 (\vec{c} = \vec{c_1} * 8 + \vec{c_2}) とスコアの一括ロード [cite: 27, 370, 400]
        // -------------------------------------------------------------
        uint8_t c1_arr[32] __attribute__((aligned(32)));
        uint8_t c2_arr[32] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)c1_arr, v_first_byte_idx);
        _mm256_store_si256((__m256i*)c2_arr, v_c2);
        
        double* cur_leaf_values = model->leaf_values[t];

        // _mm256_i32gather_pd は1度に4つのdouble要素を取るため、32サンプルを8周で処理する
        for (int i = 0; i < V_RS; i += 4) {
            // 8-bitのインデックス4つを読み込み、32-bit整数に拡張 (Zero-Extend)
            __m128i v_c1_4 = _mm_cvtsi32_si128(*(int*)&c1_arr[i]);
            __m128i v_c2_4 = _mm_cvtsi32_si128(*(int*)&c2_arr[i]);
            
            __m128i v_c1_32 = _mm_cvtepu8_epi32(v_c1_4);
            __m128i v_c2_32 = _mm_cvtepu8_epi32(v_c2_4);
            
            // インデックスの算出: \vec{c} = \vec{c_1} * 8 + \vec{c_2} 
            // (左シフト3で8倍を表現)
            __m128i v_idx = _mm_add_epi32(_mm_slli_epi32(v_c1_32, 3), v_c2_32);
            
            // Gather命令で4つのスコアを同時に取得し、現在のスコアに加算 
            __m256d v_scores = _mm256_i32gather_pd(cur_leaf_values, v_idx, 8); // scale=8(doubleのサイズ)
            
            __m256d v_curr_scores = _mm256_load_pd(&scores_v[i]);
            v_curr_scores = _mm256_add_pd(v_curr_scores, v_scores);
            _mm256_store_pd(&scores_v[i], v_curr_scores);
        }
    }
}

#ifdef __AVX512F__
static inline void rapid_scorer_avx512(RSModel *model, double **features_v, uint8_t *leaf_indexes, double *scores_v) {
    int m_bytes = (model->max_leaves + 7) / 8;
    memset(leaf_indexes, 0xFF, model->num_trees * m_bytes * V_RS_512);
    for (int k = 0; k < model->num_features; k++) {
        int eq_idx = model->feature_offsets[k];
        int next_eq_idx = model->feature_offsets[k+1];
        if (eq_idx == next_eq_idx) continue;

        __m512d f_v[8];
        for (int i = 0; i < 8; i++) f_v[i] = _mm512_set_pd(features_v[i*8+7][k], features_v[i*8+6][k], features_v[i*8+5][k], features_v[i*8+4][k], features_v[i*8+3][k], features_v[i*8+2][k], features_v[i*8+1][k], features_v[i*8+0][k]);
        
        while (eq_idx < next_eq_idx) {
            EqNode *eq = &model->eqnodes[eq_idx];
            uint64_t eta = 0;
            __m512d t_v = _mm512_set1_pd(eq->theta);
            for (int i = 0; i < 8; i++) eta |= ((uint64_t)_mm512_cmp_pd_mask(f_v[i], t_v, _CMP_LE_OQ) << (i * 8));
            
            if (eta == 0xFFFFFFFFFFFFFFFFULL) break;
            __mmask64 k_mid = eta, k_inv = ~eta;
            for (int q = 0; q < eq->u; q++) {
                int t_id = eq->tree_ids[q]; Epitome *ep = &eq->epitomes[q];
                int base = (t_id * m_bytes) * V_RS_512;
                __m512i target_fbp = _mm512_loadu_si512((__m512i*)&leaf_indexes[base + ep->fbp * V_RS_512]);
                _mm512_storeu_si512((__m512i*)&leaf_indexes[base + ep->fbp * V_RS_512], _mm512_mask_and_epi32(target_fbp, k_inv, target_fbp, _mm512_set1_epi8(ep->fb)));
                if (ep->fbp != ep->ebp) {
                    for (int b = ep->fbp + 1; b < ep->ebp; b++) {
                        __m512i target_b = _mm512_loadu_si512((__m512i*)&leaf_indexes[base + b * V_RS_512]);
                        _mm512_storeu_si512((__m512i*)&leaf_indexes[base + b * V_RS_512], _mm512_maskz_mov_epi8(k_mid, target_b));
                    }
                    __m512i target_ebp = _mm512_loadu_si512((__m512i*)&leaf_indexes[base + ep->ebp * V_RS_512]);
                    _mm512_storeu_si512((__m512i*)&leaf_indexes[base + ep->ebp * V_RS_512], _mm512_mask_and_epi32(target_ebp, k_inv, target_ebp, _mm512_set1_epi8(ep->eb)));
                }
            }
            eq_idx++;
        }
    }
    memset(scores_v, 0, sizeof(double) * V_RS_512);
    for (int t = 0; t < model->num_trees; t++) {
        uint8_t *base = &leaf_indexes[t * m_bytes * V_RS_512];
        
        uint64_t k_fin = 0;
        __m512i v_first_byte_val = _mm512_setzero_si512();
        __m512i v_first_byte_idx = _mm512_setzero_si512();

        for (int b = 0; b < m_bytes; b++) {
            __m512i v_curr_bytes = _mm512_loadu_si512((__m512i*)&base[b * V_RS_512]);
            uint64_t k_nonzero = _mm512_cmpneq_epi8_mask(v_curr_bytes, _mm512_setzero_si512());
            uint64_t k_new_found = k_nonzero & ~k_fin;
            
            v_first_byte_val = _mm512_mask_blend_epi8(k_new_found, v_first_byte_val, v_curr_bytes);
            v_first_byte_idx = _mm512_mask_blend_epi8(k_new_found, v_first_byte_idx, _mm512_set1_epi8(b));
            
            k_fin |= k_new_found;
            if (k_fin == 0xFFFFFFFFFFFFFFFFULL) break;
        }

        __m512i v_ctz_lut = _mm512_set_epi8(
            0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
            0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
            0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
            0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4
        );
        __m512i v_lo_nibble = _mm512_and_si512(v_first_byte_val, _mm512_set1_epi8(0x0F));
        __m512i v_hi_nibble = _mm512_and_si512(_mm512_srli_epi16(v_first_byte_val, 4), _mm512_set1_epi8(0x0F));
        __m512i v_ctz_lo = _mm512_shuffle_epi8(v_ctz_lut, v_lo_nibble);
        __m512i v_ctz_hi = _mm512_shuffle_epi8(v_ctz_lut, v_hi_nibble);
        uint64_t k_lo_zero = _mm512_cmpeq_epi8_mask(v_lo_nibble, _mm512_setzero_si512());
        __m512i v_c2 = _mm512_mask_blend_epi8(k_lo_zero, v_ctz_lo, _mm512_add_epi8(v_ctz_hi, _mm512_set1_epi8(4)));

        uint8_t c1_arr[64] __attribute__((aligned(64)));
        uint8_t c2_arr[64] __attribute__((aligned(64)));
        _mm512_store_si512((__m512i*)c1_arr, v_first_byte_idx);
        _mm512_store_si512((__m512i*)c2_arr, v_c2);
        
        double* cur_leaf_values = model->leaf_values[t];
        for (int i = 0; i < V_RS_512; i += 8) {
            __m256i v_idx = _mm256_add_epi32(_mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)&c1_arr[i])), 3), _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)&c2_arr[i])));
            __m512d v_scores = _mm512_i32gather_pd(v_idx, cur_leaf_values, 8);
            __m512d v_curr_scores = _mm512_loadu_pd(&scores_v[i]);
            v_curr_scores = _mm512_add_pd(v_curr_scores, v_scores);
            _mm512_storeu_pd(&scores_v[i], v_curr_scores);
        }
    }
}
#endif

#endif
