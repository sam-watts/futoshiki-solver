def dim_transform(coord, puzzle_size=5):
    """Transform a 1D grid coordinate into a 2D tuple, default 5x5 grid"""
    return coord // puzzle_size, coord % puzzle_size
