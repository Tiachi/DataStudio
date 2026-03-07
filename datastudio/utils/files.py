"""File utility functions."""

import os


def build_file(work_dir, prefix):
    """Build file path and create parent directories.

    Creates the directory structure if it doesn't exist and returns
    the full file path.

    Args:
        work_dir: Base working directory.
        prefix: Relative path including filename (can include subdirectories).

    Returns:
        str: Absolute path to the file.

    Example:
        >>> build_file('/output', 'logs/train.log')
        '/output/logs/train.log'
    """
    work_dir = os.path.abspath(work_dir)
    prefixes = prefix.split("/")
    file_name = prefixes[-1]
    prefix = "/".join(prefixes[:-1])
    if len(prefix) > 0:
        work_dir = os.path.join(work_dir, prefix)
    os.makedirs(work_dir, exist_ok=True)
    file = os.path.join(work_dir, file_name)
    return file
