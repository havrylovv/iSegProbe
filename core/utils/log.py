import io
import logging
import time
from datetime import datetime

import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

LOGGER_NAME = "root"
LOGGER_DATEFMT = "%Y-%m-%d %H:%M:%S"

handler = logging.StreamHandler()

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def add_logging(logs_path: str, prefix: str) -> None:
    log_name = (
        prefix + datetime.strftime(datetime.today(), "%Y-%m-%d_%H-%M-%S") + ".log"
    )
    stdout_log_path = logs_path / log_name

    fh = logging.FileHandler(str(stdout_log_path))
    formatter = logging.Formatter(
        fmt="(%(levelname)s) %(asctime)s: %(message)s", datefmt=LOGGER_DATEFMT
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None, mininterval=5):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.mininterval = mininterval
        self.last_time = 0

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        if len(self.buf) > 0 and time.time() - self.last_time > self.mininterval:
            self.logger.log(self.level, self.buf)
            self.last_time = time.time()


class SummaryWriterAvg(SummaryWriter):
    def __init__(self, *args, dump_period=20, **kwargs):
        super().__init__(*args, **kwargs)
        self._dump_period = dump_period
        self._avg_scalars = dict()

    def add_scalar(self, tag, value, global_step=None, disable_avg=False):
        if disable_avg or isinstance(value, (tuple, list, dict)):
            super().add_scalar(tag, np.array(value), global_step=global_step)
        else:
            if tag not in self._avg_scalars:
                self._avg_scalars[tag] = ScalarAccumulator(self._dump_period)
            avg_scalar = self._avg_scalars[tag]
            avg_scalar.add(value)

            if avg_scalar.is_full():
                super().add_scalar(tag, avg_scalar.value, global_step=global_step)
                avg_scalar.reset()


class ScalarAccumulator(object):
    def __init__(self, period):
        self.sum = 0
        self.cnt = 0
        self.period = period

    def add(self, value):
        self.sum += value
        self.cnt += 1

    @property
    def value(self):
        if self.cnt > 0:
            return self.sum / self.cnt
        else:
            return 0

    def reset(self):
        self.cnt = 0
        self.sum = 0

    def is_full(self):
        return self.cnt >= self.period

    def __len__(self):
        return self.cnt


class BufferingHandler(logging.Handler):
    """Buffer log entries and flush them to wandb"""

    def __init__(self):
        super().__init__()
        self.buffer = []

    def emit(self, record):
        self.buffer.append(self.format(record))

    def flush_to_wandb(self):
        print("#" * 80)
        print("Flushing logs to wandb")
        for log_entry in self.buffer:
            wandb.log({"log": log_entry})
        self.buffer = []


buffering_handler = BufferingHandler()
buffering_handler.setLevel(logging.INFO)
buffering_handler.setFormatter(
    logging.Formatter(
        fmt="(%(levelname)s) %(asctime)s: %(message)s", datefmt=LOGGER_DATEFMT
    )
)


def init_wandb(cfg: DictConfig) -> None:
    """Get wandb config from cfg or from file and initialize wandb"""
    if not cfg.wandb.log_wandb:
        return None
    else:
        wandb_cfg = {
            "WANDB": {
                "project": cfg.wandb.project,
                "name": cfg.wandb.name,
                "dir": cfg.wandb.dir,
            }
        }

        # make hydra cfg json serializable, to avoid errors
        cfg_json = OmegaConf.to_container(cfg, resolve=True)

        wandb.init(**wandb_cfg["WANDB"], config=cfg_json, sync_tensorboard=True)
