
# Copyright 2021 Fredrik Hallgren
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Create logging utility to be used by the rest of the code.

Creates a logger object which outputs to stdout with minimial formatting
and adds two new log levels, **header** and **subheader**, which have
additional formatting to demarcate sections of output.

Usage
~~~~~

::

    from nystrompca.utils import logger

    logger.header("Header log message")
    logger.subheader("Subheader log message")
    logger.info("Standard log message")

"""

import sys
import logging


HEADER    = 10
SUBHEADER = 15
logging.addLevelName(HEADER, 'HEADER')
logging.addLevelName(SUBHEADER, 'SUBHEADER')


def header(self, message):
    """
    New log level with customized formatting

    """
    self._log(HEADER, message, ())


def subheader(self, message):
    """
    New log level with customized formatting

    """
    self._log(SUBHEADER, message, ())


logging.Logger.header    = header
logging.Logger.subheader = subheader


class CustomFormatter(logging.Formatter):
    """
    Custom formatting for different log levels

    """

    def format(self, record):

        if record.levelno == HEADER:

            self._style._fmt = '\n %(message)s\n ' + '=' * 79

        elif record.levelno == SUBHEADER:

            self._style._fmt = '\n\n %(message)s\n ' + '-' * 79

        elif record.levelno == logging.INFO:

            self._style._fmt = '\n %(message)s'

        elif record.levelno == logging.WARN:
            self._style._fmt = '\n WARNING: %(message)s'

        elif record.levelno == logging.ERROR:
            self._style._fmt = '\n ERROR: %(message)s'

        formatted_record = logging.Formatter.format(self, record)

        return formatted_record


logger = logging.Logger('nystrompca')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(CustomFormatter())
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

