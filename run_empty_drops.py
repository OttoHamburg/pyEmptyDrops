#!/usr/bin/env python3
"""
Production-ready EmptyDrops runner script.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from empty_drops_v5_batched import empty_drops_v5_batched
