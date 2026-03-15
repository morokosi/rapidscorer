#ifndef QS_H
#define QS_H

#include <stdint.h>
#include "model.h"

typedef uint64_t bitvector;

// QuickScorer 構造体
typedef struct QSThreshold {
    double val;
    int feature_idx;
    int tree_idx;
    bitvector mask;
} QSThreshold;

typedef struct QSModel {
    QSThreshold *thresholds;
    int num_thresholds;
    int *feature_offsets;
    bitvector *init_masks;
    double **leaf_values;
    int *num_leaves;
    int num_trees;
    int num_features;
} QSModel;

// RapidScorer 構造体
typedef struct Epitome {
    uint8_t fb; uint8_t eb; uint16_t fbp; uint16_t ebp;
} Epitome;

typedef struct EqNode {
    double theta;
    int u;
    int *tree_ids;
    Epitome *epitomes;
} EqNode;

typedef struct RSModel {
    EqNode *eqnodes;
    int num_eqnodes;
    int *feature_offsets;
    double **leaf_values;
    int *num_leaves;
    int max_leaves;
    int num_trees;
    int num_features;
} RSModel;

#endif
