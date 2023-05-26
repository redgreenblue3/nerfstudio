# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Processes a video to a nerfstudio compatible dataset."""

import shutil
from dataclasses import dataclass
import json
import cv2
import csv
import numpy as np

from rich.console import Console

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import (
    ColmapConverterToNerfstudioDataset,
)

CONSOLE = Console(width=120)


@dataclass
class ProcessUnityExport(ColmapConverterToNerfstudioDataset):
    """Process videos into a nerfstudio dataset.
    This script does the following:
    1. Converts the video into images.
    2. Scales images to a specified size.
    3. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    num_frames_target: int = 300
    """Target number of frames to use for the dataset, results may not be exact."""
    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""

    def main(self) -> None:  # pylint: disable=R0915
        """Process video into a nerfstudio dataset."""

        summary_log = []
        video_path = list(self.data.glob('*.mp4'))[0]

        with open(self.data / "focal_length.txt", "r") as f:
            # Read the contents of the file
            contents = f.read()
            # Convert the contents to an integer
            focal_length = float(contents)

        cap = cv2.VideoCapture(str(video_path), 0)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        transforms_dict = {'fl_x': focal_length,
                           'fl_y': focal_length,
                           'k1': 0,
                           'k2': 0,
                           'p1': 0,
                           'p2': 0,
                           'cx': width / 2,
                           'cy': height / 2,
                           'w': width,
                           'h': height,
                           "aabb_scale": 16}

        if not self.pose_only:
            # Convert video to images
            if self.camera_type == "equirectangular":
                pass
                # create temp images folder to store the equirect and perspective images
                temp_image_dir = self.output_dir / "temp_images"
                temp_image_dir.mkdir(parents=True, exist_ok=True)
                summary_log, num_extracted_frames = process_data_utils.convert_video_to_all_images(
                    video_path=video_path,
                    image_dir=temp_image_dir,
                    crop_factor=(0.0, 0.0, 0.0, 0.0),
                    verbose=self.verbose,
                )
            else:
                pass
                summary_log, num_extracted_frames = process_data_utils.convert_video_to_all_images(
                    video_path=video_path,
                    image_dir=self.image_dir,
                    crop_factor=self.crop_factor,
                    verbose=self.verbose,
                )

            # Generate planar projections if equirectangular
            if self.camera_type == "equirectangular":
                perspective_image_size = equirect_utils.compute_resolution_from_equirect(
                    self.output_dir / "temp_images", self.images_per_equirect
                )

                equirect_utils.generate_planar_projections_from_equirectangular(
                    self.output_dir / "temp_images",
                    perspective_image_size,
                    self.images_per_equirect,
                    crop_factor=self.crop_factor,
                )

                # copy the perspective images to the image directory
                process_data_utils.copy_images(
                    self.output_dir / "temp_images" / "planar_projections",
                    image_dir=self.output_dir / "images",
                    verbose=False,
                )

                # remove the temp_images folder
                shutil.rmtree(self.output_dir / "temp_images", ignore_errors=True)

                self.camera_type = "perspective"

            # Create mask
            mask_path = process_data_utils.save_mask(
                image_dir=self.image_dir,
                num_downscales=self.num_downscales,
                crop_factor=(0.0, 0.0, 0.0, 0.0),
                percent_radius=self.percent_radius_crop,
            )
            if mask_path is not None:
                summary_log.append(f"Saved mask to {mask_path}")

            # Downscale images
            summary_log.append(
                process_data_utils.downscale_images(self.image_dir, self.num_downscales, verbose=self.verbose)
            )

            # Export depth maps
            image_id_to_depth_path, log_tmp = self._export_depth()
            summary_log += log_tmp

            summary_log += self._save_transforms(num_extracted_frames, image_id_to_depth_path, mask_path)

        frames = []

        with open(self.data / "poses.csv", "r") as csvfile:
            csvreader = csv.reader(csvfile)

            for row in csvreader:
                frame_id = int(row[0]) + 1
                file_path = './images/frame_%05d.png' % frame_id
                transform_matrix = [[float(row[j+1]) for j in range(4*i, 4*i+4)] for i in range(4)]
                transform_matrix = np.array(transform_matrix)
                # transform_matrix[:,2] = -transform_matrix[:,2]

                transform_matrix[:3, :3] = np.linalg.inv(transform_matrix[:3, :3])
                transform_matrix[:3, 3] = np.matmul(transform_matrix[:3, :3], -transform_matrix[:3, 3])
                axis_permutation = np.array([[1, 0, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 0, 1]])
                transform_matrix = np.matmul(axis_permutation, transform_matrix)

                transform_matrix = transform_matrix.tolist()
                frame = {'file_path': file_path, 'transform_matrix': transform_matrix}
                frames.append(frame)

        transforms_dict["frames"] = frames

        with open(self.output_dir / "transforms.json", "w") as f:
            json.dump(transforms_dict, f, indent=4, sort_keys=False)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        if not self.poses_only:
            for summary in summary_log:
                CONSOLE.log(summary)