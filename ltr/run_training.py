import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.admin.settings as ws_settings


def run_training(train_module, train_name, execution_name=None, cudnn_benchmark=True, start_epoch=None):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name if execution_name is None else f'{train_name}_{execution_name}')
    settings.start_epoch = start_epoch

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    print("Num devices: ", torch.cuda.device_count())
    print("Mem info: ", torch.tensor(torch.cuda.mem_get_info())/(1024**3))
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('execution_name', type=str, help='Name of the run.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--start_epoch', type=str, help='Epoch number of checkpoint to start from (0-N) or "last" for latest checkpoint; default="last"', default='last')

    args = parser.parse_args()

    start_epoch = None if args.start_epoch=="last" else int(args.start_epoch)

    run_training(args.train_module, args.train_name, args.execution_name, args.cudnn_benchmark, start_epoch)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
