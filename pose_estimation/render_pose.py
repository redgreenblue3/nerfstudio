# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated

from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_path_from_json,
    get_spiral_path,
)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn

from nerfstudio.scripts.render import *

import argparse
import wandb
import torchvision.transforms.functional as TF

run = wandb.init(
  project="nerfstudio-project",
  notes="My first experiment",
  tags=["baseline", "paper1"]
)

parser = argparse.ArgumentParser()
parser.add_argument('--load_config')
parser.add_argument('--camera_path_filename')
args = parser.parse_args()

# Get config (model) and cameras paths
eval_num_rays_per_chunk: Optional[int] = None
load_config = Path(args.load_config)
camera_path_filename = Path(args.camera_path_filename)

# Get pipeline (mostly, trained nerf model)
_, pipeline, _, _ = eval_setup(
    load_config,
    eval_num_rays_per_chunk=eval_num_rays_per_chunk,
    test_mode="inference",
)

# Get camera information
with open(camera_path_filename, "r", encoding="utf-8") as f:
    camera_path = json.load(f)
seconds = camera_path["seconds"]
crop_data = get_crop_from_json(camera_path)
cameras = get_path_from_json(camera_path)
cameras = cameras.to(pipeline.device)

aabb_box = None
camera_ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=aabb_box)

with torch.no_grad():
    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

image = outputs['rgb']
imp = image.permute(2,0,1)
im = TF.to_pil_image(imp)
wimage = wandb.Image(im)
wandb.log({"image": wimage}, commit=True)

pass
