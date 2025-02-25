import functools
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import haiku as hk
import jax
import jax.experimental.maps
import jax.numpy as jnp
from jax import config, tree_util
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint as pjit_sharding_constraint
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as P
