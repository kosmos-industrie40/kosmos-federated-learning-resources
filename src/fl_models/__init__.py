"""
Module imports
"""
# -*- coding: utf-8 -*-
# from pkg_resources import DistributionNotFound, get_distribution
#
# try:
#    # Change here if project is renamed and does not equal the package name
#    dist_name = "federated-bearing-use-case"
#    __version__ = get_distribution(dist_name).version
# except DistributionNotFound:
#    __version__ = "unknown"
# finally:
#    del get_distribution, DistributionNotFound
__version__ = "0.0.1"

from . import util
from . import abstract
from . import cnn_model
from . import data_set_type
from . import ffnn
