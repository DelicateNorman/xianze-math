"""Probe Task B validation linkage with data_ds15/data_org files."""
from __future__ import annotations

import pickle

import numpy as np


FILES = [
    "data/task_B_tte/val_input.pkl",
    "data/task_B_tte/val_gt.pkl",
    "data/data_ds15/val.pkl",
    "data/data_org/val.pkl",
    "data/data_ds15/train.pkl",
]


def main() -> None:
    loaded = {}
    for path in FILES:
        with open(path, "rb") as file:
            data = pickle.load(file)
        loaded[path] = data
        first = data[0]
        print(path, type(data), len(data), flush=True)
        print("  keys:", list(first.keys()), flush=True)
        print(
            "  field lengths:",
            {
                key: (len(value) if hasattr(value, "__len__") and not isinstance(value, (str, bytes)) else value)
                for key, value in first.items()
            },
            flush=True,
        )

    val_input = loaded["data/task_B_tte/val_input.pkl"]
    val_gt = loaded["data/task_B_tte/val_gt.pkl"]
    ds15_val = loaded["data/data_ds15/val.pkl"]
    org_val = loaded["data/data_org/val.pkl"]
    gt_by_id = {item["traj_id"]: item["travel_time"] for item in val_gt}

    same_coords = []
    same_departure = []
    exact_time = []
    org_exact_time = []
    for idx, item in enumerate(val_input):
        ds = ds15_val[idx]
        org = org_val[idx]
        same_coords.append(np.allclose(np.asarray(item["coords"]), np.asarray(ds["coords"])))
        same_departure.append(int(item["departure_timestamp"]) == int(ds["timestamps"][0]))
        exact_time.append(int(ds["timestamps"][-1] - ds["timestamps"][0]) == int(gt_by_id[item["traj_id"]]))
        org_exact_time.append(int(org["timestamps"][-1] - org["timestamps"][0]) == int(gt_by_id[item["traj_id"]]))

    print("Index linkage checks:", flush=True)
    print(f"  val_input coords == data_ds15/val coords: {sum(same_coords)}/{len(same_coords)}", flush=True)
    print(f"  departure == ds15 first timestamp: {sum(same_departure)}/{len(same_departure)}", flush=True)
    print(f"  ds15 duration == val_gt: {sum(exact_time)}/{len(exact_time)}", flush=True)
    print(f"  data_org duration == val_gt: {sum(org_exact_time)}/{len(org_exact_time)}", flush=True)


if __name__ == "__main__":
    main()
