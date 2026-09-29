#!/usr/bin/env python
# coding: utf-8

# **Wir befinden uns hier in einem Python 3.8, aber 3DINO ist mit Python 3.9 gebastelt, also müssen wir testen, ob es überhaupt funktioniert.**  
# Anpassungen:
# * Wörkaround für `args.arch.removesuffix` in `dinov2/models/__init__.py`

# In[1]:


# Changes introduced to DINO to make it work with our dataset and on Python3.8
# get_ipython().system(' cd DINO; git diff')


# In[2]:


import sys
from pathlib import Path
import subprocess
import pandas as pd
import json
import os
from datetime import datetime


# In[3]:


# Projekt Konstanten

PROJECT_DIR = Path("/data_rechenknecht03_2/students/boehmerle")
OUTPUT_DIR = PROJECT_DIR / "outputs"
REPO_DIR = PROJECT_DIR / "DINO" # Python Packages können nicht mit einer Zahl anfangen, also nutzen wir nur DINO, auch wenn es eigentlich 3DINO ist.

DATASET_NAME = "hip_ultrasound"
DATALIST_DIR = PROJECT_DIR / "datalists"

CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "3dino_vit_weights.pth"

MAX_WORKERS = min(8, os.cpu_count() or 1)


# In[4]:


# 3DINO klonen und importieren

if not REPO_DIR.exists():
    subprocess.run(
        [
            "git", "clone",
            "https://github.com/AICONSlab/3DINO.git",
            str(REPO_DIR),
        ],
        check=True,
    )
else:
    print(f"Repository existiert bereits: {REPO_DIR}")

commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

print(f"3DINO-Commit: {commit}")

import DINO.dinov2


# In[5]:


# Datensatz für 3DINO vorbereiten
DATALIST_DIR.mkdir(exist_ok=True)

manifest_path = OUTPUT_DIR / "manifest_with_split.csv"
manifest = pd.read_csv(manifest_path)

datalist = {}

for split_name in ["training", "validation", "test"]:
    subset = (
        manifest
        .query("pair_status == 'ok' and split == @split_name")
        .sort_values(["case_id", "side", "sweep_id"])
    )

    datalist[split_name] = [
        {
            "image": row.image_path,
            "label": row.label_path,
        }
        for row in subset.itertuples(index=False)
    ]

datalist_path = DATALIST_DIR / f"{DATASET_NAME}_100_datalist.json"

with open(datalist_path, "w", encoding="utf-8") as file:
    json.dump(datalist, file, indent=2)

print(f"Gespeichert: {datalist_path}")
for split_name, entries in datalist.items():
    print(f"{split_name}: {len(entries)} Sweeps")

print("\nErster Trainingseintrag:")
print(json.dumps(datalist["training"][0], indent=2))


# In[6]:


def run_segmentation(
    output_dir,
    batch_size=1,
    num_workers=MAX_WORKERS,
    segmentation_head="Linear",
    epochs=1,
    epoch_length=1,
    learning_rate=1e-4):

    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(REPO_DIR)
    run_env["CUDA_VISIBLE_DEVICES"] = "0"
    # run_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Ist wohl hier leider nicht verfügbar.
    # Eigentlich keine gute Idee, aber in diesem Fall wird uns der Log vollgespammt.
    run_env["PYTHONWARNINGS"] = "ignore:You are using `torch.load` with `weights_only=False`:FutureWarning"

    full_output_dir = OUTPUT_DIR / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = full_output_dir / "run.log"
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "dinov2/eval/segmentation3d.py",
                "--config-file", "dinov2/configs/train/vit3d_highres.yaml",
                "--dataset-name", "hip_ultrasound",
                "--dataset-percent", "100",
                "--base-data-dir", str(DATALIST_DIR),
                "--pretrained-weights", str(CHECKPOINT_PATH),
                "--segmentation-head", "Linear",
                "--image-size", "96",
                "--resize-scale", "1.0",
                "--eval-iters", str(max(1, int(epoch_length / 2))),
                "--warmup-iters", "0",
                "--segmentation-head", segmentation_head,
                "--epochs", str(epochs),
                "--epoch-length", str(epoch_length),
                "--batch-size", str(batch_size),
                "--num-workers", str(num_workers),
                "--learning-rate", str(learning_rate),
                "--output-dir", str(full_output_dir),
                "--cache-dir", str(PROJECT_DIR / "cache" / "hip_ultrasound"),
            ],
            cwd=REPO_DIR,
            text=True,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1
        )

        for line in iter(process.stdout.readline, ""):
            print(f"[{datetime.now().time()}]", line, end="")
            log_file.write(line)
            log_file.flush()

        process.stdout.close()

        return [process.wait(), log_file]


# In[ ]:


# Tut das ganze überhaut?
# [return_code, log_path] = run_segmentation("hip_ultrasound_smoke_linear", batch_size=1, epochs=1, epoch_length=1, learning_rate=1e-4)

# print(f"Finished at {datetime.now().time()}")
# print(f"Return-Code: {return_code}")
# print(f"Logdatei: {log_path}")


# In[ ]:


# Ein größerer Test mit etwas mehr Daten und mehr Epochen:
[return_code, log_path] = run_segmentation("hip_ultrasound_smoke_linear", batch_size=5, epochs=100, epoch_length=10, learning_rate=1e-4)

print(f"Finished at {datetime.now().time()}")
print(f"Return-Code: {return_code}")
print(f"Logdatei: {log_path}")

