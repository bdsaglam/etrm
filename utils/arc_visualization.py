"""Grid visualization utilities for ARC puzzles."""

from typing import Optional, Tuple

import numpy as np

# ARC color palette (standard 10 colors)
ARC_COLORS = [
    [0, 0, 0],  # 0: black
    [0, 116, 217],  # 1: blue
    [255, 65, 54],  # 2: red
    [46, 204, 64],  # 3: green
    [255, 220, 0],  # 4: yellow
    [170, 170, 170],  # 5: gray
    [240, 18, 190],  # 6: magenta
    [255, 133, 27],  # 7: orange
    [127, 219, 255],  # 8: cyan
    [135, 12, 37],  # 9: maroon
]


def get_content_bounds(
    flat_grid: np.ndarray, grid_size: int = 30
) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounds of actual content (colors 2-11) in a flat grid.

    Args:
        flat_grid: 1D array of tokens (0=PAD, 1=EOS, 2-11=colors)
        grid_size: Grid dimension (default 30)

    Returns:
        Tuple of (row_start, row_end, col_start, col_end) or None if no content
    """
    grid = flat_grid.reshape(grid_size, grid_size)
    mask = (grid >= 2) & (grid <= 11)

    if not mask.any():
        return None

    rows_with_content = mask.any(axis=1)
    cols_with_content = mask.any(axis=0)

    row_indices = np.where(rows_with_content)[0]
    col_indices = np.where(cols_with_content)[0]

    return (row_indices[0], row_indices[-1] + 1, col_indices[0], col_indices[-1] + 1)


def grid_to_image(grid: np.ndarray, cell_size: int = 8) -> np.ndarray:
    """Convert ARC grid to RGB image.

    Args:
        grid: 2D array of color values (0-9)
        cell_size: Size of each cell in pixels

    Returns:
        RGB image array of shape (H*cell_size, W*cell_size, 3)
    """
    h, w = grid.shape
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            color_idx = int(grid[i, j]) % len(ARC_COLORS)
            color = ARC_COLORS[color_idx]
            img[
                i * cell_size : (i + 1) * cell_size, j * cell_size : (j + 1) * cell_size
            ] = color

    return img


def flat_grid_to_2d(
    flat_grid: np.ndarray,
    grid_size: int = 30,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    crop: bool = True,
) -> np.ndarray:
    """Convert flattened grid (with padding/EOS) to 2D grid.

    Args:
        flat_grid: 1D array of length seq_len (e.g., 900)
        grid_size: Maximum grid dimension
        bounds: Optional (row_start, row_end, col_start, col_end) to crop to.
                If None and crop=True, auto-detects content bounds.
        crop: If True, crop to content bounds. If False, return full grid.

    Returns:
        2D grid (cropped to content if crop=True, full grid otherwise)
    """
    # Reshape to 2D
    grid = flat_grid.reshape(grid_size, grid_size)

    # Convert token values to colors (tokens 2-11 are colors 0-9)
    # Token 0 = PAD, Token 1 = EOS
    grid = np.clip(grid.astype(np.int32) - 2, 0, 9)

    # Return full grid if cropping disabled
    if not crop:
        return grid

    # Detect bounds if not provided
    if bounds is None:
        bounds = get_content_bounds(flat_grid, grid_size)

    # Return full grid if bounds are still missing
    if bounds is None:
        return grid

    row_start, row_end, col_start, col_end = bounds
    return grid[row_start:row_end, col_start:col_end]
