import numpy as np
import lightgbm as lgb
import os
import json
import struct
from sklearn.datasets import load_svmlight_file

def load_data(file_path):
    X, y, qid = load_svmlight_file(file_path, query_id=True)
    group = []
    if len(qid) > 0:
        curr_qid = qid[0]
        curr_count = 0
        for q in qid:
            if q == curr_qid:
                curr_count += 1
            else:
                group.append(curr_count)
                curr_qid = q
                curr_count = 1
        group.append(curr_count)
    return X, y, np.array(group, dtype=np.int32)

def process_node(node, nodes_list):
    node_idx = len(nodes_list)
    nodes_list.append(None)
    if "leaf_index" in node:
        nodes_list[node_idx] = {"is_leaf": 1, "leaf_value": node["leaf_value"]}
    else:
        left_idx = process_node(node["left_child"], nodes_list)
        right_idx = process_node(node["right_child"], nodes_list)
        nodes_list[node_idx] = {
            "is_leaf": 0, "feature_idx": node["split_feature"],
            "threshold": node["threshold"], "left_idx": left_idx, "right_idx": right_idx
        }
    return node_idx

def export_binary(model_json, output_path):
    trees = model_json["tree_info"]
    num_trees = len(trees)
    num_features = model_json["max_feature_idx"] + 1
    with open(output_path, "wb") as f:
        f.write(struct.pack("=ii", num_trees, num_features))
        for tree in trees:
            nodes_list = []
            process_node(tree["tree_structure"], nodes_list)
            num_nodes = len(nodes_list)
            f.write(struct.pack("=i", num_nodes))
            for node in nodes_list:
                f.write(struct.pack("=i", node["is_leaf"]))
                if node["is_leaf"]:
                    f.write(struct.pack("=d", node["leaf_value"]))
                    f.write(struct.pack("=iff", 0, 0, 0))
                else:
                    f.write(struct.pack("=idii", node["feature_idx"], node["threshold"], node["left_idx"], node["right_idx"]))

def main():
    train_path = "MSLR-WEB10K/Fold1/train.txt"
    vali_path = "MSLR-WEB10K/Fold1/vali.txt"
    X_train, y_train, g_train = load_data(train_path)
    X_vali, y_vali, g_vali = load_data(vali_path)
    
    train_ds = lgb.Dataset(X_train, label=y_train, group=g_train)
    vali_ds = lgb.Dataset(X_vali, label=y_vali, group=g_vali, reference=train_ds)

    tree_range = [10, 100, 500, 1000]
    leaf_range = [16, 32, 64]

    os.makedirs("models", exist_ok=True)

    for n_trees in tree_range:
        for n_leaves in leaf_range:
            print(f"Training: Trees={n_trees}, Leaves={n_leaves}")
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'num_leaves': n_leaves,
                'learning_rate': 0.1,
                'num_iterations': n_trees,
                'verbose': -1
            }
            model = lgb.train(params, train_ds, valid_sets=[vali_ds], 
                              callbacks=[lgb.log_evaluation(period=0)])
            
            bin_path = f"models/model_t{n_trees}_l{n_leaves}.bin"
            export_binary(model.dump_model(), bin_path)
            # Also save standard text for baseline C API
            model.save_model(f"models/model_t{n_trees}_l{n_leaves}.txt")

if __name__ == "__main__":
    main()
