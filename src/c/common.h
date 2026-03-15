#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

typedef struct {
    double *features;
    float label;
} Document;

typedef struct {
    Document *docs;
    int num_docs;
    int num_features;
} Dataset;

static inline Dataset* load_dataset(const char *filename, int max_rows, int num_features) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;

    Dataset *ds = malloc(sizeof(Dataset));
    ds->docs = malloc(sizeof(Document) * max_rows);
    ds->num_features = num_features;
    ds->num_docs = 0;

    char line[16384];
    while (ds->num_docs < max_rows && fgets(line, sizeof(line), fp)) {
        Document *doc = &ds->docs[ds->num_docs];
        doc->features = calloc(num_features, sizeof(double));
        
        char *ptr = line;
        char *endptr;
        doc->label = strtof(ptr, &endptr);
        ptr = endptr;

        // Skip qid
        while (*ptr && *ptr != ' ') ptr++;
        while (*ptr == ' ') ptr++;

        while (*ptr && *ptr != '\n' && *ptr != '\r') {
            int feat_idx = (int)strtol(ptr, &endptr, 10);
            if (ptr == endptr) break;
            ptr = endptr;
            if (*ptr == ':') {
                ptr++;
                double val = strtod(ptr, &endptr);
                if (feat_idx >= 1 && feat_idx <= num_features) {
                    doc->features[feat_idx - 1] = val;
                }
                ptr = endptr;
            }
            while (*ptr == ' ') ptr++;
        }
        ds->num_docs++;
    }
    fclose(fp);
    return ds;
}

static inline void free_dataset(Dataset *ds) {
    if (!ds) return;
    for (int i = 0; i < ds->num_docs; i++) {
        free(ds->docs[i].features);
    }
    free(ds->docs);
    free(ds);
}

static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#endif
