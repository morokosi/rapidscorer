#ifndef QS_CONVERSION_H
#define QS_CONVERSION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "qs.h"

// Helper to get leaf masks
static void build_leaf_masks(Node *nodes, int curr_idx, int *leaf_ptr, int target_node_idx, bitvector *out_mask, uint8_t *out_mask_bytes, int is_in_target_left) {
    if (nodes[curr_idx].is_leaf) {
        if (is_in_target_left) {
            if (out_mask) *out_mask &= ~(1ULL << (*leaf_ptr));
            if (out_mask_bytes) out_mask_bytes[(*leaf_ptr) / 8] &= ~(1 << ((*leaf_ptr) % 8));
        }
        (*leaf_ptr)++;
        return;
    }
    int in_left = is_in_target_left || (curr_idx == target_node_idx);
    build_leaf_masks(nodes, nodes[curr_idx].split.left_child_idx, leaf_ptr, target_node_idx, out_mask, out_mask_bytes, in_left);
    build_leaf_masks(nodes, nodes[curr_idx].split.right_child_idx, leaf_ptr, target_node_idx, out_mask, out_mask_bytes, is_in_target_left);
}

// --- QuickScorer ---
typedef struct RawQSNode { double val; int feature_idx; int tree_idx; int node_idx; } RawQSNode;
static int compare_qs_raw(const void *a, const void *b) {
    RawQSNode *na = (RawQSNode *)a; RawQSNode *nb = (RawQSNode *)b;
    if (na->feature_idx != nb->feature_idx) return na->feature_idx - nb->feature_idx;
    return (na->val < nb->val) ? -1 : 1;
}

static QSModel* convert_to_qs(Model *model) {
    QSModel *qs = malloc(sizeof(QSModel));
    qs->num_trees = model->num_trees;
    qs->num_features = model->num_features;
    qs->num_leaves = malloc(sizeof(int) * qs->num_trees);
    qs->leaf_values = malloc(sizeof(double*) * qs->num_trees);
    qs->init_masks = malloc(sizeof(bitvector) * qs->num_trees);
    int total_int = 0;
    for (int i = 0; i < model->num_trees; i++) {
        int l = 0; for (int j = 0; j < model->trees[i].num_nodes; j++) if (model->trees[i].nodes[j].is_leaf) l++; else total_int++;
        qs->num_leaves[i] = l; qs->init_masks[i] = (l == 64) ? ~0ULL : (1ULL << l) - 1;
        qs->leaf_values[i] = malloc(sizeof(double) * l);
        int li = 0;
        void collect(Node *ns, int c, int tree_id, double **lvs, int *li_ptr) {
            if (ns[c].is_leaf) lvs[tree_id][(*li_ptr)++] = ns[c].leaf_value;
            else { collect(ns, ns[c].split.left_child_idx, tree_id, lvs, li_ptr); collect(ns, ns[c].split.right_child_idx, tree_id, lvs, li_ptr); }
        }
        collect(model->trees[i].nodes, 0, i, qs->leaf_values, &li);
    }
    RawQSNode *raw = malloc(sizeof(RawQSNode) * total_int);
    int curr_n = 0;
    for (int i = 0; i < model->num_trees; i++) {
        for (int j = 0; j < model->trees[i].num_nodes; j++) {
            if (!model->trees[i].nodes[j].is_leaf) {
                raw[curr_n].val = model->trees[i].nodes[j].split.threshold;
                raw[curr_n].feature_idx = model->trees[i].nodes[j].split.feature_idx;
                raw[curr_n].tree_idx = i;
                raw[curr_n].node_idx = j;
                curr_n++;
            }
        }
    }
    qsort(raw, total_int, sizeof(RawQSNode), compare_qs_raw);
    qs->num_thresholds = total_int;
    qs->thresholds = malloc(sizeof(QSThreshold) * total_int);
    qs->feature_offsets = malloc(sizeof(int) * (qs->num_features + 1));
    memset(qs->feature_offsets, -1, sizeof(int) * (qs->num_features + 1));
    for (int i = 0; i < total_int; i++) {
        if (qs->feature_offsets[raw[i].feature_idx] == -1) qs->feature_offsets[raw[i].feature_idx] = i;
        qs->thresholds[i].val = raw[i].val;
        qs->thresholds[i].feature_idx = raw[i].feature_idx;
        qs->thresholds[i].tree_idx = raw[i].tree_idx;
        qs->thresholds[i].mask = ~0ULL;
        int leaf_ptr = 0;
        build_leaf_masks(model->trees[raw[i].tree_idx].nodes, 0, &leaf_ptr, raw[i].node_idx, &qs->thresholds[i].mask, NULL, 0);
    }
    free(raw); return qs;
}

static double qs_score(QSModel *model, double *features, bitvector *v) {
    for (int i = 0; i < model->num_trees; i++) v[i] = model->init_masks[i];
    for (int k = 0; k < model->num_features; k++) {
        int idx = model->feature_offsets[k]; if (idx == -1) continue;
        int next_feat_idx = model->num_thresholds;
        for (int next_k = k + 1; next_k < model->num_features; next_k++) {
            if (model->feature_offsets[next_k] != -1) { next_feat_idx = model->feature_offsets[next_k]; break; }
        }
        while (idx < next_feat_idx) {
            QSThreshold *t = &model->thresholds[idx];
            if (features[t->feature_idx] > t->val) v[t->tree_idx] &= t->mask;
            else break;
            idx++;
        }
    }
    double s = 0;
    for (int i = 0; i < model->num_trees; i++) s += model->leaf_values[i][__builtin_ctzll(v[i])];
    return s;
}

// --- RapidScorer ---
typedef struct RawRSNode { double theta; int feature_idx; int tree_id; int node_idx; } RawRSNode;
static int compare_rs_raw(const void *a, const void *b) {
    RawRSNode *na = (RawRSNode *)a; RawRSNode *nb = (RawRSNode *)b;
    if (na->feature_idx != nb->feature_idx) return na->feature_idx - nb->feature_idx;
    return (na->theta < nb->theta) ? -1 : 1;
}

static RSModel* convert_to_rs(Model *model) {
    RSModel *rs = malloc(sizeof(RSModel));
    rs->num_trees = model->num_trees;
    rs->num_features = model->num_features;
    rs->num_leaves = malloc(sizeof(int) * rs->num_trees);
    rs->leaf_values = malloc(sizeof(double*) * rs->num_trees);
    int max_l = 0, total_int = 0;
    for (int i = 0; i < model->num_trees; i++) {
        int l = 0; for (int j = 0; j < model->trees[i].num_nodes; j++) if (model->trees[i].nodes[j].is_leaf) l++; else total_int++;
        rs->num_leaves[i] = l; if (l > max_l) max_l = l;
        rs->leaf_values[i] = malloc(sizeof(double) * l);
        int li = 0;
        void collect(Node *ns, int c, int tree_id, double **lvs, int *li_ptr) {
            if (ns[c].is_leaf) lvs[tree_id][(*li_ptr)++] = ns[c].leaf_value;
            else { collect(ns, ns[c].split.left_child_idx, tree_id, lvs, li_ptr); collect(ns, ns[c].split.right_child_idx, tree_id, lvs, li_ptr); }
        }
        collect(model->trees[i].nodes, 0, i, rs->leaf_values, &li);
    }
    rs->max_leaves = max_l;
    int m_bytes = (max_l + 7) / 8;
    RawRSNode *raw = malloc(sizeof(RawRSNode) * total_int);
    int curr_n = 0;
    for (int i = 0; i < model->num_trees; i++) {
        for (int j = 0; j < model->trees[i].num_nodes; j++) {
            if (!model->trees[i].nodes[j].is_leaf) {
                raw[curr_n].theta = model->trees[i].nodes[j].split.threshold;
                raw[curr_n].feature_idx = model->trees[i].nodes[j].split.feature_idx;
                raw[curr_n].tree_id = i;
                raw[curr_n].node_idx = j;
                curr_n++;
            }
        }
    }
    qsort(raw, total_int, sizeof(RawRSNode), compare_rs_raw);
    rs->eqnodes = malloc(sizeof(EqNode) * total_int);
    rs->feature_offsets = malloc(sizeof(int) * (rs->num_features + 1));
    int num_u = 0;
    int curr_feat = 0;
    for (int i = 0; i < total_int; ) {
        int s = i; 
        int feat_idx = raw[s].feature_idx;
        while (curr_feat <= feat_idx) {
            rs->feature_offsets[curr_feat++] = num_u;
        }

        while (i < total_int && raw[i].feature_idx == feat_idx && raw[i].theta == raw[s].theta) i++;
        int u = i - s; EqNode *eq = &rs->eqnodes[num_u]; eq->theta = raw[s].theta; eq->u = u;
        eq->tree_ids = malloc(sizeof(int) * u); eq->epitomes = malloc(sizeof(Epitome) * u);
        for (int j = 0; j < u; j++) {
            RawRSNode *rn = &raw[s+j]; eq->tree_ids[j] = rn->tree_id;
            uint8_t *mask = malloc(m_bytes); memset(mask, 0xFF, m_bytes);
            int leaf_ptr = 0;
            build_leaf_masks(model->trees[rn->tree_id].nodes, 0, &leaf_ptr, rn->node_idx, NULL, mask, 0);
            int fbp = -1, ebp = -1; for (int b = 0; b < m_bytes; b++) if (mask[b] != 0xFF) { if (fbp == -1) fbp = b; ebp = b; }
            if (fbp == -1) { eq->epitomes[j].fb = 0xFF; eq->epitomes[j].eb = 0xFF; eq->epitomes[j].fbp = 0; eq->epitomes[j].ebp = 0; }
            else { eq->epitomes[j].fb = mask[fbp]; eq->epitomes[j].eb = mask[ebp]; eq->epitomes[j].fbp = fbp; eq->epitomes[j].ebp = ebp; }
            free(mask);
        }
        num_u++;
    }
    while (curr_feat <= rs->num_features) {
        rs->feature_offsets[curr_feat++] = num_u;
    }
    rs->num_eqnodes = num_u; free(raw); return rs;
}

static void free_qs_model(QSModel *qs) {
    if (!qs) return;
    for (int i = 0; i < qs->num_trees; i++) free(qs->leaf_values[i]);
    free(qs->leaf_values); free(qs->num_leaves); free(qs->thresholds); free(qs->init_masks); free(qs->feature_offsets); free(qs);
}

static void free_rs_model(RSModel *rs) {
    if (!rs) return;
    for (int i = 0; i < rs->num_eqnodes; i++) { free(rs->eqnodes[i].tree_ids); free(rs->eqnodes[i].epitomes); }
    free(rs->eqnodes); for (int i = 0; i < rs->num_trees; i++) free(rs->leaf_values[i]);
    free(rs->leaf_values); free(rs->num_leaves); free(rs->feature_offsets); free(rs);
}

#endif
