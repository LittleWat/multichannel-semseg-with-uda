import json

import numpy as np
import pandas as pd

with open("nyu_gt_ditribution.json") as f:
    nyu_dist = pd.DataFrame(pd.Series(json.load(f)), columns=["nyu"])

with open("/home/mil/watanabe/Git/DomainAdaptation/VisDA2017/segmentation/suncg_gt_distribution.json") as f:
    suncg_dist = pd.DataFrame(pd.Series(json.load(f)), columns=["suncg"])

nyu_dist.index = nyu_dist.index.astype(np.int64)
suncg_dist.index = suncg_dist.index.astype(np.int64)

dist_df = nyu_dist.join(suncg_dist)
dist_df = dist_df.fillna(0)

class_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refridgerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop"
]
class_names.append("background")

dist_df["cls_name"] = class_names
dist_df.index.name = "id"
dist_df.to_csv("nyu_suncg_distribution.csv")

# plot
dist_df_without_background = dist_df.head(40)
dist_df_without_background["nyu_prob"] = dist_df_without_background["nyu"] / dist_df_without_background["nyu"].sum()
dist_df_without_background["suncg_prob"] = dist_df_without_background["suncg"] / dist_df_without_background[
    "suncg"].sum()
dist_df_without_background.index = dist_df_without_background.cls_name

ax = dist_df_without_background.sort_index()[["nyu_prob", "suncg_prob"]].plot(kind="bar", color=["orange", "steelblue"])
ax.set_xlabel('')
ax.set_ylabel('Ratio of pixels')
fig = ax.get_figure()
fig.savefig("nyu_suncg_distribution.png", bbox_inches='tight', pad_inches=0, dpi=300)
