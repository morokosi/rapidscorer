#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

typedef struct Node {
    int is_leaf;
    union {
        struct {
            int feature_idx;
            double threshold;
            int left_child_idx;
            int right_child_idx;
        } split;
        double leaf_value;
    };
} Node;

typedef struct Tree {
    Node *nodes;
    int num_nodes;
} Tree;

typedef struct Model {
    Tree *trees;
    int num_trees;
    int num_features;
} Model;

#endif
