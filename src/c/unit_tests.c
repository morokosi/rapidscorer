#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "model_util.h"
#include "qs.h"
#include "qs_conversion.h"
#include "vqs_rs_impl.h"

// Helper to create a simple 2-tree model manually
Model* create_simple_model() {
    Model *m = malloc(sizeof(Model));
    m->num_trees = 2;
    m->num_features = 2;
    m->trees = malloc(sizeof(Tree) * 2);

    // Tree 0: if f0 <= 0.5: 1.0 else: 2.0
    m->trees[0].num_nodes = 3;
    m->trees[0].nodes = malloc(sizeof(Node) * 3);
    m->trees[0].nodes[0].is_leaf = 0;
    m->trees[0].nodes[0].split.feature_idx = 0;
    m->trees[0].nodes[0].split.threshold = 0.5f;
    m->trees[0].nodes[0].split.left_child_idx = 1;
    m->trees[0].nodes[0].split.right_child_idx = 2;
    m->trees[0].nodes[1].is_leaf = 1;
    m->trees[0].nodes[1].leaf_value = 1.0;
    m->trees[0].nodes[2].is_leaf = 1;
    m->trees[0].nodes[2].leaf_value = 2.0;

    // Tree 1: if f1 <= 10.0: 0.5 else: 1.5
    m->trees[1].num_nodes = 3;
    m->trees[1].nodes = malloc(sizeof(Node) * 3);
    m->trees[1].nodes[0].is_leaf = 0;
    m->trees[1].nodes[0].split.feature_idx = 1;
    m->trees[1].nodes[0].split.threshold = 10.0f;
    m->trees[1].nodes[0].split.left_child_idx = 1;
    m->trees[1].nodes[0].split.right_child_idx = 2;
    m->trees[1].nodes[1].is_leaf = 1;
    m->trees[1].nodes[1].leaf_value = 0.5;
    m->trees[1].nodes[2].is_leaf = 1;
    m->trees[1].nodes[2].leaf_value = 1.5;

    return m;
}

void test_qs() {
    printf("Testing QuickScorer... ");
    Model *raw = create_simple_model();
    QSModel *qs = convert_to_qs(raw);
    bitvector *v = malloc(sizeof(bitvector) * qs->num_trees);

    double f1[2] = {0.0, 5.0}; // Expected: 1.0 + 0.5 = 1.5
    assert(fabs(qs_score(qs, f1, v) - 1.5) < 1e-12);

    double f2[2] = {1.0, 15.0}; // Expected: 2.0 + 1.5 = 3.5
    assert(fabs(qs_score(qs, f2, v) - 3.5) < 1e-12);

    free(v); free_qs_model(qs); free_model(raw);
    printf("PASS\n");
}

void test_vqs() {
    printf("Testing V-QuickScorer... ");
    Model *raw = create_simple_model();
    QSModel *qs = convert_to_qs(raw);
    
    bitvector *v_v_flat = malloc(sizeof(bitvector) * V_QS * qs->num_trees);
    bitvector **v_v = malloc(sizeof(bitvector*) * V_QS);
    for(int i=0; i<V_QS; i++) v_v[i] = &v_v_flat[i * qs->num_trees];
    double **feats = malloc(sizeof(double*) * 4);
    double scores[4];

    for(int i=0; i<4; i++) feats[i] = malloc(sizeof(double) * 2);
    
    feats[0][0] = 0.0; feats[0][1] = 5.0;  // 1.5
    feats[1][0] = 1.0; feats[1][1] = 5.0;  // 2.5
    feats[2][0] = 0.0; feats[2][1] = 15.0; // 2.5
    feats[3][0] = 1.0; feats[3][1] = 15.0; // 3.5
    
    vqs_score_avx2(qs, feats, v_v, scores);
    assert(fabs(scores[0] - 1.5) < 1e-12);
    assert(fabs(scores[1] - 2.5) < 1e-12);
    assert(fabs(scores[2] - 2.5) < 1e-12);
    assert(fabs(scores[3] - 3.5) < 1e-12);

    for(int i=0; i<4; i++) free(feats[i]);
    free(feats); free(v_v_flat); free(v_v); free_qs_model(qs); free_model(raw);
    printf("PASS\n");
}

void test_rs() {
    printf("Testing RapidScorer... ");
    Model *raw = create_simple_model();
    RSModel *rs = convert_to_rs(raw);
    
    uint8_t *leaf_idxs = malloc(rs->num_trees * ((rs->max_leaves+7)/8) * 32);
    double **feats = malloc(sizeof(double*) * 32);
    double scores[32];

    for(int i=0; i<32; i++) {
        feats[i] = malloc(sizeof(double) * 136);
        if (i % 4 == 0) { feats[i][0] = 0.0; feats[i][1] = 5.0; }
        else if (i % 4 == 1) { feats[i][0] = 1.0; feats[i][1] = 5.0; }
        else if (i % 4 == 2) { feats[i][0] = 0.0; feats[i][1] = 15.0; }
        else { feats[i][0] = 1.0; feats[i][1] = 15.0; }
    }

    rapid_scorer_avx2(rs, feats, leaf_idxs, scores);
    for(int i=0; i<32; i++) {
        if (i % 4 == 0) assert(fabs(scores[i] - 1.5) < 1e-12);
        else if (i % 4 == 1) assert(fabs(scores[i] - 2.5) < 1e-12);
        else if (i % 4 == 2) assert(fabs(scores[i] - 2.5) < 1e-12);
        else assert(fabs(scores[i] - 3.5) < 1e-12);
    }

    for(int i=0; i<32; i++) free(feats[i]);
    free(feats); free(leaf_idxs); free_rs_model(rs); free_model(raw);
    printf("PASS\n");
}

int main() {
    test_qs();
    test_vqs();
    test_rs();
    printf("All unit tests passed successfully!\n");
    return 0;
}
