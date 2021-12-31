import argparse


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym Experiment Manager')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='path to configuration file')
    parser.add_argument('--repeat', type=int, default=1,
                        help='number of repeating jobs')
    parser.add_argument('--mark_done', action='store_true',
                        help='mark yaml as done after a job has finished')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='remaining options (see graphgym/config.py)')

    return parser.parse_args()
