from collections import namedtuple
from dataclasses import dataclass
from random import choice, randint, random, sample
from typing import Any, List, Tuple

import numpy as np
from pandas.compat._optional import import_optional_dependency

from .dataset import BANDS_GROUPS_IDX

MASK_STRATEGIES = (
    "group_bands",
    "random_timesteps",
    "chunk_timesteps",
    "random_combinations",
)


def make_mask(strategy: str, mask_ratio: float, num_timesteps: int,
              bands_groups_idx: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a mask for a given strategy and percentage of masked values.
    Args:
        strategy: The masking strategy to use. One of MASK_STRATEGIES
        mask_ratio: The percentage of values to mask. Between 0 and 1.
    """
    TIMESTEPS_IDX = list(range(num_timesteps))
    BAND_EXPANSION = [len(x) for x in bands_groups_idx.values()]

    mask = np.full((num_timesteps, len(bands_groups_idx)), False)
    dw_mask = np.full(num_timesteps, False)
    num_tokens_to_mask = int(
        ((num_timesteps * len(bands_groups_idx)) + 1) * mask_ratio)

    # print(f"Num tokens to mask: {num_tokens_to_mask}")

    # def mask_topography(srtm_mask, num_tokens_to_mask, mask_ratio):
    #     should_flip = random() < mask_ratio
    #     if should_flip:
    #         srtm_mask = True
    #         num_tokens_to_mask -= 1
    #     return srtm_mask, num_tokens_to_mask

    def random_masking(mask, dw_mask, num_tokens_to_mask: int):
        if num_tokens_to_mask > 0:
            eo_tokens_mask = mask.flatten()
            unmasked_tokens = eo_tokens_mask == False
            idx = np.flatnonzero(unmasked_tokens)
            np.random.shuffle(idx)
            idx = idx[:num_tokens_to_mask]
            eo_tokens_mask[idx] = True
            mask = eo_tokens_mask.reshape(
                (num_timesteps, len(bands_groups_idx)))

            # dw_tokens_mask = dw_mask.flatten()
            # unmasked_tokens = dw_tokens_mask == False
            # idx = np.flatnonzero(unmasked_tokens)
            # np.random.shuffle(idx)
            # idx = idx[:num_tokens_to_mask]
            # dw_tokens_mask[idx] = True
            # dw_mask = dw_tokens_mask.reshape((num_timesteps))
            # dw_mask = all_tokens_mask[:num_timesteps]
        return mask, dw_mask

    # RANDOM BANDS
    if strategy == "random_combinations":
        # srtm_mask, num_tokens_to_mask = mask_topography(
        #     srtm_mask, num_tokens_to_mask, mask_ratio)
        mask, dw_mask = random_masking(mask, dw_mask, num_tokens_to_mask)

    elif strategy == "group_bands":
        # srtm_mask, num_tokens_to_mask = mask_topography(
        #     srtm_mask, num_tokens_to_mask, mask_ratio)
        # next, we figure out how many tokens we can mask
        num_band_groups_to_mask = int(num_tokens_to_mask / num_timesteps)
        num_tokens_to_mask -= num_timesteps * num_band_groups_to_mask
        assert num_tokens_to_mask >= 0
        # tuple because of mypy, which thinks lists can only hold one type
        band_groups: List[Any] = list(range(len(bands_groups_idx))) + ["LC"]
        band_groups_to_mask = sample(band_groups, num_band_groups_to_mask)
        for band_group in band_groups_to_mask:
            if band_group == "LC":
                dw_mask[:] = True
            else:
                mask[:, band_group] = True
        mask, dw_mask = random_masking(mask, dw_mask, num_tokens_to_mask)

    # RANDOM TIMESTEPS
    elif strategy == "random_timesteps":
        # srtm_mask, num_tokens_to_mask = mask_topography(
        #     srtm_mask, num_tokens_to_mask, mask_ratio)
        # +1 for dynamic world, -1 for the SRTM
        timesteps_to_mask = int(num_tokens_to_mask / (len(bands_groups_idx)))
        num_tokens_to_mask -= (len(bands_groups_idx)) * timesteps_to_mask
        timesteps = sample(TIMESTEPS_IDX, k=timesteps_to_mask)
        mask[timesteps] = True
        dw_mask[timesteps] = True
        # mask, dw_mask = random_masking(mask, dw_mask, num_tokens_to_mask)
    elif strategy == "chunk_timesteps":
        # srtm_mask, num_tokens_to_mask = mask_topography(
        #     srtm_mask, num_tokens_to_mask, mask_ratio)
        timesteps_to_mask = int(num_tokens_to_mask / (len(bands_groups_idx)))
        num_tokens_to_mask -= (len(bands_groups_idx)) * timesteps_to_mask
        start_idx = randint(0, num_timesteps - timesteps_to_mask)
        mask[start_idx:start_idx + timesteps_to_mask] = True  # noqa
        dw_mask[start_idx:start_idx + timesteps_to_mask] = True  # noqa
        mask, dw_mask = random_masking(mask, dw_mask, num_tokens_to_mask)
    else:
        raise ValueError(
            f"Unknown strategy {strategy} not in {MASK_STRATEGIES}")

    return np.repeat(mask, BAND_EXPANSION, axis=1), dw_mask


@dataclass
class MaskParams:
    strategies: Tuple[str, ...] = ("NDVI", )
    ratio: float = 0.5

    def __post_init__(self):
        for strategy in self.strategies:
            assert strategy in [
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ]

    def mask_data(
        self, eo_data: np.ndarray, lc_data: np.ndarray,
        lc_missing_data_class: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, str]:
        strategy = choice(self.strategies)
        mask, lc_mask = make_mask(strategy=strategy,
                                  mask_ratio=self.ratio,
                                  num_timesteps=eo_data.shape[0],
                                  bands_groups_idx=BANDS_GROUPS_IDX)
        x = eo_data * ~mask
        y = np.zeros(eo_data.shape).astype(np.float32)
        y[mask] = eo_data[mask]

        masked_lc_tokens = np.ones_like(lc_data) * lc_missing_data_class
        x_lc = np.where(lc_mask, masked_lc_tokens, lc_data)
        y_lc = np.zeros(x_lc.shape).astype(np.int16)
        y_lc[lc_mask] = lc_data[lc_mask]

        return mask, lc_mask, x, y, x_lc, y_lc, strategy
