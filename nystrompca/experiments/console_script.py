
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
Console script entry point to run the different experiments

"""

import argparse

from nystrompca.experiments import (methods_experiments, bound_experiments,
                                    regression_experiments)
from nystrompca.utils import logger


def main():
    """
    Entry point for command-line utility.

    Parses the arguments supplied to the command-line script and runs
    the selected experiment. Captures any exception that occurs and
    just prints the error message.

    """
    description = "Run the different Nyström kernel PCA experiments."
    epilog="Display subcommand options with nystrompca <subcommand> -h"
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog)
    subparsers = parser.add_subparsers(required=True,
                                       dest='subcommand',
                                       title="available subcommands",
                                       help="which experiment to run")

    methods_description = methods_experiments.parser.description
    subparser_methods = subparsers.add_parser('methods',
                                       description= methods_description,
                                       parents=[methods_experiments.parser],
                                       add_help=False)
    subparser_methods.set_defaults(func=methods_experiments.main)

    bound_description = bound_experiments.parser.description
    subparser_bound = subparsers.add_parser('bound',
                                       description=bound_description,
                                       parents=[bound_experiments.parser],
                                       add_help=False)
    subparser_bound.set_defaults(func=bound_experiments.main)

    reg_description = regression_experiments.parser.description
    subparser_reg = subparsers.add_parser('regression',
                                       description=reg_description,
                                       parents=[regression_experiments.parser],
                                       add_help=False)
    subparser_reg.set_defaults(func=regression_experiments.main)

    args = vars(parser.parse_args())

    args.pop('subcommand')

    func = args.pop('func')

    try:
        func(**args)
    except Exception as e: # pylint: disable=broad-except
        logger.error(str(e))

