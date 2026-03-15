#ifndef MODEL_UTIL_H
#define MODEL_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include "model.h"

Model* load_model(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    Model *model = malloc(sizeof(Model));
    if (fread(&model->num_trees, sizeof(int), 1, fp) != 1) { fclose(fp); free(model); return NULL; }
    if (fread(&model->num_features, sizeof(int), 1, fp) != 1) { fclose(fp); free(model); return NULL; }
    model->trees = malloc(sizeof(Tree) * model->num_trees);

    for (int i = 0; i < model->num_trees; i++) {
        Tree *tree = &model->trees[i];
        if (fread(&tree->num_nodes, sizeof(int), 1, fp) != 1) break;
        tree->nodes = malloc(sizeof(Node) * tree->num_nodes);
        for (int j = 0; j < tree->num_nodes; j++) {
            Node *node = &tree->nodes[j];
            if (fread(&node->is_leaf, sizeof(int), 1, fp) != 1) break;
            if (node->is_leaf) {
                if (fread(&node->leaf_value, sizeof(double), 1, fp) != 1) break;
                float dummy[3];
                if (fread(dummy, sizeof(float), 3, fp) != 3) break;
            } else {
                if (fread(&node->split.feature_idx, sizeof(int), 1, fp) != 1) break;
                if (fread(&node->split.threshold, sizeof(double), 1, fp) != 1) break;
                if (fread(&node->split.left_child_idx, sizeof(int), 1, fp) != 1) break;
                if (fread(&node->split.right_child_idx, sizeof(int), 1, fp) != 1) break;
            }
        }
    }
    fclose(fp);
    return model;
}

static inline void free_model(Model *model) {
    if (!model) return;
    for (int i = 0; i < model->num_trees; i++) {
        free(model->trees[i].nodes);
    }
    free(model->trees);
    free(model);
}

#endif
