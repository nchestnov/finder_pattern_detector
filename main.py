import argparse
from pathlib import Path
import time

from src.finder_pattern_detector import FinderPatternDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Dataset name", type=str)
    args = parser.parse_args()

    dataset_path = Path.cwd() / "input" / args.dataset

    det = FinderPatternDetector()

    img_paths = list(dataset_path.glob('**/*'))
    time_start = time.time()
    for path in img_paths:
        det.detect(path, output_img_path=Path.cwd() / "output" / args.dataset, verbose=False)
        print(f"Image:\t{str(path)}")
    time_spent = time.time() - time_start
    print("=" * 50)
    print(f"Time spent:\t{time_spent:.02f} seconds\t({(time_spent / len(img_paths)):.02f} sec/image)")


if __name__ == '__main__':
    main()
