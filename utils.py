import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_path_abs(origin_path):
    return os.path.join(BASE_DIR, origin_path)
