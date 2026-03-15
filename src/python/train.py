import numpy as np
import lightgbm as lgb
import os
import json
from sklearn.datasets import load_svmlight_file

def load_data(file_path, max_rows=100000):
    print(f"Loading {file_path} (max {max_rows} rows)...")
    # We can't easily limit rows in load_svmlight_file without reading the whole thing or using a temporary file
    # For now, let's use a temporary file with first max_rows
    tmp_file = "tmp_subset.txt"
    os.system(f"head -n {max_rows} {file_path} > {tmp_file}")
    
    # Remove qid: for LightGBM if we were to use native, but sklearn likes it.
    # However, sklearn load_svmlight_file handles qid:X if query_id=True
    X, y, qid = load_svmlight_file(tmp_file, query_id=True)
    os.remove(tmp_file)
    
    # Compute group counts from qid
    # qid is usually sorted in MSLR-WEB10K
    _, counts = np.unique(qid, return_counts=True)
    # If qid is not sorted, unique(return_counts) might be wrong for ranking if groups are interleaved
    # But in MSLR they are contiguous. Let's be safe and count contiguous.
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

def main():
    train_data_path = "MSLR-WEB10K/Fold1/train.txt"
    vali_data_path = "MSLR-WEB10K/Fold1/vali.txt"
    
    X_train, y_train, g_train = load_data(train_data_path, 10000)
    X_vali, y_vali, g_vali = load_data(vali_data_path, 2000)
    
    train_dataset = lgb.Dataset(X_train, label=y_train, group=g_train)
    vali_dataset = lgb.Dataset(X_vali, label=y_vali, group=g_vali, reference=train_dataset)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': '1,3,5,10',
        'learning_rate': 0.1,
        'num_leaves': 64,
        'min_data_in_leaf': 100,
        'num_iterations': 50 # Small number of trees for faster inference testing initially
    }
    
    print("Training model...")
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[vali_dataset],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    print("Saving model...")
    model.save_model("model.txt")
    
    print("Dumping model to JSON for easy parsing in C...")
    model_json = model.dump_model()
    with open("model.json", "w") as f:
        json.dump(model_json, f)

if __name__ == "__main__":
    main()
