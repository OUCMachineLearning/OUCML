import argparse
import yaml
import os
import logging

logger = logging.getLogger()

class Config(object):
    def __init__(self, filename=None):
        assert os.path.exists(filename), "ERROR: Config File doesn't exist."
        try:
            with open(filename, 'r') as f:
                self._cfg_dict = yaml.load(f)
        # parent of IOError, OSError *and* WindowsError where available
        except EnvironmentError:
            logger.error('Please check the file with name of "%s"', filename)
        logger.info(' APP CONFIG '.center(80, '-'))
        logger.info(''.center(80, '-'))

    def __getattr__(self, name):
        value = self._cfg_dict[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value