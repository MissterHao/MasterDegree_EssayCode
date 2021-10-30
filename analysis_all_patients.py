import os 
import pandas as pd
from AnalysisJob import AnalysisJob

target_dir = "image_data"
origin_dir = os.path.join(target_dir, "original")
label_dir = os.path.join(target_dir, "label")

origin_files = os.listdir(origin_dir)
label_files = os.listdir(label_dir)


# weights = [1, 1, 1, 1]
# weights = [0.81665246, 0.28122385, 0.36095798, 0.15496431]
weights = [0.9151752, 0.52506548, 0.14795733, 0.20648379]

info: pd.DataFrame = pd.read_excel("./info_v3.xlsx")
info["weights"] = None
info["F1-Score"] = 0
info["weights"].astype(object)

for index, filename in enumerate(info["filename"]):
# for filename in origin_files:
    # 確保label和original都有相同的檔案名稱
    if filename not in label_files:
        continue

    task = AnalysisJob(filename)
    task.upload_weights(weights)
    token = task.getHistoryToken(note="")

    # 前處理
    task.preprocessing()

    # Mask
    task.findMask()

    # Vote
    mine_f1   = task.vote(weights)
    normal_f1 = task.vote([1, 1, 1, 1])
    voter_result = task.refVote()

    info.loc[info.filename==filename, "weights"][0] = weights
    

    # 4個比較用的 voter
    info.loc[info.filename==filename, "QuickShift"] = voter_result[0]
    info.loc[info.filename==filename, "Felzenszwalb"] = voter_result[1]
    info.loc[info.filename==filename, "SLIC"] = voter_result[2]
    info.loc[info.filename==filename, "Watershed"] = voter_result[3]

    # 我的方法
    info.loc[info.filename==filename, "F1-Score"] = mine_f1

    # 1 1 1 1 正常的
    info.loc[info.filename==filename, "Normal F1-Score"] = normal_f1



    print(f"{index:>3}", f" {filename:>35}", mine_f1, normal_f1, voter_result)

    # Upload
    task.uploadResult(token)


info.to_excel(f"info_v3_w{'_'.join([str(w) for w in weights])}.xlsx", index=False, sheet_name="整理檔案資訊")