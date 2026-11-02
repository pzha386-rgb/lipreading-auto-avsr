#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os

import cv2
import numpy as np
# from skimage import transform as tf   # <-- removed


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


def warp_img(src, dst, img, std_size):
    """
    OpenCV replacement for:
        tform = tf.estimate_transform("similarity", src, dst)
        warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    We estimate a 2x3 affine (similarity-like) transform with OpenCV and warp.
    - src, dst: (N, 2) arrays of corresponding points
    - img: HxWxC uint8 (or HxW)
    - std_size: (H_out, W_out)
    Returns:
        warped_image (uint8), transform_matrix (2x3)
    """
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)

    # Estimate affine (scale+rotation+translation) mapping src -> dst
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    # cv2 expects dsize as (width, height); std_size is (H, W)
    warped = cv2.warpAffine(
        img,
        M,
        dsize=(int(std_size[1]), int(std_size[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Ensure uint8 output to match prior behavior
    if warped.dtype != np.uint8:
        warped = np.clip(warped, 0, 255).astype(np.uint8)

    return warped, M


def apply_transform(transform, img, std_size):
    """
    OpenCV replacement for:
        warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    Here, `transform` is the 2x3 matrix mapping input -> output coordinates.
    """
    warped = cv2.warpAffine(
        img,
        transform,
        dsize=(int(std_size[1]), int(std_size[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if warped.dtype != np.uint8:
        warped = np.clip(warped, 0, 255).astype(np.uint8)
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    """
    Crops a centered face patch of fixed size (2*height, 2*width) from `img`
    around the mean of `landmarks`. If the crop would exceed the image
    boundaries, it shifts the window to stay fully inside.
    """
    h, w = img.shape[:2]
    center_x, center_y = np.mean(landmarks, axis=0)

    # Check for too much bias (optional)
    # if abs(center_y - h / 2) > height + threshold:
    #     raise OverflowError("too much bias in height")
    # if abs(center_x - w / 2) > width + threshold:
    #     raise OverflowError("too much bias in width")

    # Desired top-left corner
    x_min = int(round(center_x - width))
    y_min = int(round(center_y - height))

    # Ensure the crop stays within bounds by shifting if needed
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0

    x_max = x_min + 2 * width
    y_max = y_min + 2 * height

    # Shift window back inside the image if it overflows
    if x_max > w:
        x_max = w
        x_min = w - 2 * width
    if y_max > h:
        y_max = h
        y_min = h - 2 * height

    # Final clipping in case of rounding issues
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # Extract the patch
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])

    return cutted_img


class VideoProcess:
    def __init__(
        self,
        mean_face_path="20words_mean_face.npy",
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
        convert_gray=True,
    ):
        self.reference = np.load(
            os.path.join(os.path.dirname(__file__), mean_face_path)
        )
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

    def __call__(self, video, landmarks):
        # Pre-process landmarks: interpolate frames that are not detected
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        # Exclude corner cases: no landmark in all frames or number of frames is less than window length
        if (
            not preprocessed_landmarks
            or len(preprocessed_landmarks) < self.window_margin
        ):
            return
        # Affine transformation and crop patch
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None, "crop an empty patch."
        return sequence

    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            window_margin = min(
                self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx
            )
            smoothed_landmarks = np.mean(
                [
                    landmarks[x]
                    for x in range(
                        frame_idx - window_margin, frame_idx + window_margin + 1
                    )
                ],
                axis=0,
            )
            smoothed_landmarks += landmarks[frame_idx].mean(
                axis=0
            ) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference, grayscale=self.convert_gray
            )
            patch = cut_patch(
                transformed_frame,
                transformed_landmarks[self.start_idx : self.stop_idx],
                self.crop_height // 2,
                self.crop_width // 2,
            )
            sequence.append(patch)
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(
                    landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
                )

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle corner case: keep frames at the beginning or at the end that failed to be detected
        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [
                landmarks[valid_frames_idx[0]]
            ] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks

    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=True,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    ):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(
            reference, stable_points, reference_size, target_size
        )
        transform = self.estimate_affine_transform(
            landmarks, stable_points, stable_reference
        )
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame,
            landmarks,
            transform,
            target_size,
            interpolation,
            border_mode,
            border_value,
        )

        return transformed_frame, transformed_landmarks

    def get_stable_reference(
        self, reference, stable_points, reference_size, target_size
    ):
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]

    def apply_affine_transform(
        self,
        frame,
        landmarks,
        transform,
        target_size,
        interpolation,
        border_mode,
        border_value,
    ):
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(int(target_size[0]), int(target_size[1])),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = (
            np.matmul(landmarks, transform[:, :2].transpose())
            + transform[:, 2].transpose()
        )
        return transformed_frame, transformed_landmarks
