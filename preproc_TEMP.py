"""
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import gc
import numbers
import shutil
import torch
import json
from tqdm import trange, tqdm
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
)
import torchvision.transforms._functional_video as F
from typing import Dict

import wget

import matplotlib.pyplot as plt

_RUN_SIZE = 5
_BATCH_SIZE = 1000


DEVICE = "cpu"
print(f"Using device: {DEVICE}")
DTYPE = "float32"


FILEPATH = "drive/MyDrive/video_arsenals/"

PITCH_TYPES = ["FF", "SI", "CH", "CB", "FV", "SL"]
PITCH_INDICATORS = [f"is_{pitch_type}" for pitch_type in PITCH_TYPES]
PITCH_FEATURES = [
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "release_pos_y",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "release_spin_rate",
    "release_extension",
]
PITCH_FEATURES_DELTA = [item + "_FF_delta" for item in PITCH_FEATURES]

FEATURES = PITCH_FEATURES + PITCH_INDICATORS  # PITCH_FEATURES_DELTA
TARGET = "delta_run_exp"
KEY_COLS = [
    "pitcher",
    "batter",
    "game_pk",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
]


def download_mp4(url, filepath=None):
    """"""
    file_name = wget.download(url, filepath)
    return file_name


def save_pickle(file, filename):
    """
    Pickles something
    """
    with open(filename, "wb") as handle:
        pickle.dump(file, handle)


def load_pickle(filename):
    """
    Reads back in a pickled object
    """
    with open(filename, "rb") as handle:
        result = pickle.load(handle)
    return


class LeftCropVideo:
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            # presented (H, W)
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, clip, offset=25):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        # https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_video.py
        return F.crop(clip, i=0, j=offset, h=self.crop_size[0], w=self.crop_size[1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"


def make_x3d_transform(model_name="x3d_m"):
    """ """
    assert model_name in ("x3d_m", "x3d_s", "x3d_xs")

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    frames_per_second = 30
    model_transform_params = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        },
    }

    # Get transform parameters based on model
    transform_params = model_transform_params[model_name]

    # Note that this transform is specific to the slow_R50 model.
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                LeftCropVideo(
                    crop_size=(
                        transform_params["crop_size"],
                        transform_params["crop_size"],
                    )
                ),
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (
        transform_params["num_frames"] * transform_params["sampling_rate"]
    ) / frames_per_second
    return transform, clip_duration


def make_slowfast_transform():
    ####################
    # SlowFast transform
    ####################

    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
    alpha = 4

    class PackPathway(torch.nn.Module):
        """
        Transform for converting video frames as a list of tensors.
        """

        def __init__(self):
            super().__init__()

        def forward(self, frames: torch.Tensor):
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
            return frame_list

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                LeftCropVideo(crop_size),
                PackPathway(),
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate) / frames_per_second
    return transform, clip_duration


def retrieve_model(model_name):
    """"""
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", model=model_name, pretrained=True
    )

    # Set to eval mode and move to desired device
    model = model.to(DEVICE)
    model = model.eval()
    return model


def retrieve_slowfast_embeddings(model, video, start_sec=0.0, max_sec=5.0):
    """"""
    # setup helper functions and clips
    transform, clip_duration = make_slowfast_transform()

    # outer results
    # results, ctr = [], 1
    input_1, input_2, ctr = [], [], 1

    # read in as many videos as specified, and get embeddings
    while start_sec <= max_sec:
        # Load the desired clip
        video_data = video.get_clip(
            start_sec=start_sec, end_sec=start_sec + clip_duration
        )
        # Apply a transform to normalize the video input
        video_data = transform(video_data)
        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = [i.to("cpu")[None, ...] for i in inputs]

        # compute model embeddings
        # results.append(model(inputs))

        input_1.append(inputs[0])
        input_2.append(inputs[1])

        start_sec += clip_duration
        ctr += 1

    # compile all inputs
    input_1 = torch.stack(input_1).squeeze(1).to(DEVICE)
    input_2 = torch.stack(input_2).squeeze(1).to(DEVICE)
    # churn through model
    return (
        model([input_1, input_2]).flatten().unsqueeze(0).to("cpu").detach().numpy(),
        (input_1.to("cpu").detach().numpy(), input_2.to("cpu").detach().numpy()),
    )


def retrieve_x3d_embeddings(model, video, start_sec=0.0, max_sec=5.0):
    """"""
    # setup helper functions and clips
    transform, clip_duration = make_x3d_transform()

    # outer results
    inputs_tall, ctr = [], 1

    # read in as many videos as specified, and get embeddings
    while start_sec <= max_sec:
        # Load the desired clip
        video_data = video.get_clip(
            start_sec=start_sec, end_sec=start_sec + clip_duration
        )
        # Apply a transform to normalize the video input
        video_data = transform(video_data)
        # Move the inputs to the desired device
        inputs = video_data["video"].to("cpu")[None, ...]
        inputs_tall.append(inputs)

        start_sec += clip_duration
        ctr += 1
    inputs_tall = torch.stack(inputs_tall).squeeze(1).to(DEVICE)
    # churn through model
    return (
        model(inputs_tall).flatten().unsqueeze(0).to("cpu").detach().numpy(),
        inputs_tall.to("cpu").detach().numpy(),
    )


def get_args():
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx")
    args = parser.parse_args()
    return int(args.start_idx)


if __name__ == "__main__":

    start_idx = get_args()
    print(f"Using starting index: {start_idx}")

    _FEATURES_SLOWFAST = {}
    _FEATURES_X3D = {}

    ### setup a temp folder
    foldername = "videos_TEMP"
    if foldername in os.listdir():
        shutil.rmtree(foldername)
    os.mkdir(foldername)

    ### setup features
    keys, y, x_pitch = [], [], []
    ### setup tracking items
    slowfast_emds, x3d_emds, inputs = [], [], []

    ### fetch models
    model_x3d = retrieve_model("x3d_m")
    model_slowfast = retrieve_model("slowfast_r50")

    ### make hooks to extract conv features from models
    def get_features_x3d(name):
        def hook(model, input, output):
            _FEATURES_X3D[name] = output.to("cpu").detach().numpy()

        return hook

    def get_features_slowfast(name):
        def hook(model, input, output):
            _FEATURES_SLOWFAST[name] = output.to("cpu").detach().numpy()

        return hook

    # after final dropout
    model_slowfast.blocks[6].dropout.register_forward_hook(
        get_features_slowfast("emds")
    )
    # (N, 2048, 1, 2, 2) // relu before final dropout
    model_x3d.blocks[5].pool.post_act.register_forward_hook(get_features_x3d("emds"))

    data_df = pd.read_csv("/home/ec2-user/data_video.csv")
    data_df = data_df.sample(data_df.shape[0], replace=False, random_state=605400).iloc[
        start_idx : start_idx + _RUN_SIZE, :
    ].reset_index(drop=True)
    os.remove("/home/ec2-user/data_video.csv")

    slowfast_emds, x3d_emds, inputs = [], [], []

    num_batches_processed, ctr = 1, 1

    for i, df in tqdm(data_df.iterrows()):
        print(f"Iter: {i}/{data_df.shape[0]}")
        try:
            # download and load video locally
            video_path = download_mp4(df["video_url"], foldername)
            video = EncodedVideo.from_path(video_path)

            # slowfast embeddings
            emd_sf_iter, _ = retrieve_slowfast_embeddings(
                model_slowfast, video, start_sec=0.0, max_sec=5.0
            )
            slowfast_emds.append(_FEATURES_SLOWFAST["emds"])


            emd_x3d_iter, inputs_x3d_iter = retrieve_x3d_embeddings(
                model_x3d, video, start_sec=0.0, max_sec=5.0
            )
            x3d_emds.append(_FEATURES_X3D["emds"])
            # inputs.append(inputs_x3d_iter)

            # gather pitch features
            x_pitch.append(torch.tensor(df[FEATURES].values.astype(DTYPE)))
            # gather response
            y.append(torch.tensor(df[TARGET]))
            # gather keys
            keys.append(df[KEY_COLS])

            os.remove(video_path)
            gc.collect()

            if ctr == _BATCH_SIZE:
                np.savez_compressed(f"x3d_emds_{start_idx}_{num_batches_processed}", np.stack(x3d_emds), )
                np.savez_compressed(f"slowfast_emds_{start_idx}_{num_batches_processed}", np.stack(slowfast_emds), )
                # np.savez_compressed(np.stack(inputs), f"inputs_{start_idx}_{num_batches_processed}")
                np.savez_compressed(f"y_{start_idx}_{num_batches_processed}", np.stack(y), )
                np.savez_compressed(f"x_pitch_{start_idx}_{num_batches_processed}", np.stack(x_pitch), )
                np.savez_compressed(f"keys_{start_idx}_{num_batches_processed}", keys,)

                num_batches_processed += 1  # increment index
                ctr = 1  # reset counter

                inputs, y, x_pitch, keys = [], [], [], []
            else:
                ctr += 1

            # file cleanup
            for file in os.listdir():
                if "mp4" in file:
                    os.remove(file)
                    print(" --dropping: {file}")
        except:
            print(
                f"Failure: game_pk={df['game_pk']}, at_bat_number={df['at_bat_number']}, pitch_number={df['pitch_number']}"
            )

    np.savez_compressed(f"x3d_emds_{start_idx}_{num_batches_processed}", np.stack(x3d_emds), )
    np.savez_compressed(f"slowfast_emds_{start_idx}_{num_batches_processed}", np.stack(slowfast_emds), )
    # np.savez_compressed(np.stack(inputs), f"inputs_{start_idx}_{num_batches_processed}")
    np.savez_compressed(f"y_{start_idx}_{num_batches_processed}", np.stack(y), )
    np.savez_compressed(f"x_pitch_{start_idx}_{num_batches_processed}", np.stack(x_pitch), )
    np.savez_compressed(f"keys_{start_idx}_{num_batches_processed}", keys,)
