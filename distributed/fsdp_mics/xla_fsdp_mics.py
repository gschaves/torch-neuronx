# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
#

from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
import torch_xla.core.xla_model as xm

import os
from dataclasses import dataclass
from typing import List, Optional, Union
import torch


@dataclass
class MiCSShardingConfig:
    """
    Sharding configuration for MiCS:
    - Each partition group maintains a copy of the entire model states, and
    - devices in a replication group hold the same part of model states

    For example, suppose we have two nodes and each node has 32 ranks, if we
    set the partition world size to 32, then
    - partition_groups = [[0,...,31], [32,...,63]]
    - partition_rank is the local rank of each partition group
      could be 0 ... 31 in this example
    - partition_group_size = 32
    - replication_groups = [[0,32], [1,33], ..., [31,63]]
    - replication_group_size = 2
    """

    partition_groups: List[List[int]]
    partition_rank: int
    partition_group_size: int
    replication_groups: List[List[int]]
    replication_group_size: int


class XlaFullyShardedDataParallelMiCS(XlaFullyShardedDataParallel):
    """
    A wrapper for using MiCS on top of FSDP.

    Pseudo-code usage::

    from torch_neuronx.distributed.fsdp_mics import XlaFullyShardedDataParallelMiCS as FSDP
    mics_module = FSDP(my_module)
    optim = torch.optim.Adam(mics_module.parameters(), lr=0.0001)
    output = mics_module(x, y)
    loss = output.sum()
    loss.backward()
    mics_module.mics_gradient_sync()
    mics_module.clip_grad_norm_(max_norm=1)
    optim.step()

    The only usage difference between the MiCS and exisiting FSDP:
    - replace XlaFullyShardedDataParallel with XlaFullyShardedDataParallelMiCS
    - add an env var for partition_group_size if not using the default value 32
    - add mics_module.mics_gradient_sync() to reduce gradients across replicas

    MiCS paper:
    - publised at VLDB'22
    - https://www.amazon.science/publications/mics-near-linear-scaling-for-training-gigantic-model-on-public-cloud  # noqa: E501

    MiCS idea is shown as follow:
    - Instead of using all devices as a single group for holding the model states, MiCS divides all devices into
      partition groups. Each group holds a complete copy of the model states. Within each group, the model states
      are partitioned. Thus most frequent parameter gatherings are operated at the scale of each partition group;
    - Unlike ZeRO that synchronizes gradients over all devices for each micro-step, MiCS only synchronizes gradients
      within the partition group until the gradient accumulation boundary is reached. At the gradient accumulation
      boundary, gradients are synchronized at the scale of each replication group (across the partition groups).
      Devices in a replication group hold the same part of model states.
    """

    def __init__(self, *args, **kwargs):
        # initialize MiCS sharding config
        self.mics_sharding_cfg = self.init_mics_sharding_cfg()

        # add sharding_groups, sharding_rank and sharding_world_size into fsdp params
        fsdp_sharding_params = self.init_fsdp_sharding_args(kwargs)
        # initialize FSDP
        XlaFullyShardedDataParallel.__init__(self, *args, **fsdp_sharding_params)

        if self._debug_print:
            xm.master_print(
                f"MiCS sharding conifg:\n"
                f"\tpartition_groups={self.mics_sharding_cfg.partition_groups}\n"
                f"\tpartition_rank={self.mics_sharding_cfg.partition_rank}\n"
                f"\tpartition_group_size={self.mics_sharding_cfg.partition_group_size}\n"
                f"\treplication_groups={self.mics_sharding_cfg.replication_groups}\n"
                f"\treplication_group_size={self.mics_sharding_cfg.replication_group_size}",
                flush=True,
            )

    def init_mics_sharding_cfg(self):
        world_size = xm.xrt_world_size()
        rank = xm.get_ordinal()

        # set partition_group_size to NEURON_MICS_PARTITION_GROUP_SIZE or default value
        partition_group_size = int(
            os.environ.get("NEURON_MICS_PARTITION_GROUP_SIZE", 32)
        )
        # sanity check for partition_group_size
        if partition_group_size <= 0 or (
            partition_group_size not in {2, 8} and partition_group_size % 32 != 0
        ):
            raise ValueError(
                f"partition_group_size({partition_group_size}) not supported. \
                  Supported partition_group_size: 2, 8, positive multiples of 32."
            )
        if world_size % partition_group_size != 0:
            raise ValueError(
                f"world_size({world_size}) should be a multiple of partition_group_size({partition_group_size})."  # noqa: E501
            )

        # partition_rank is a local rank within a partition group used by FSDP
        partition_rank = rank % partition_group_size

        # replication_group_size equals to the length of partition_groups
        replication_group_size = world_size // partition_group_size

        # suppose we have two nodes and each node has 32 ranks,
        # if we set the partition world size to 32, then
        # partition_groups = [[0,...,31], [32,...,63]
        partition_groups = []
        for i in range(0, replication_group_size):
            partition_groups.append(
                [
                    j
                    for j in range(
                        i * partition_group_size, (i + 1) * partition_group_size
                    )
                ]
            )

        # suppose we have two nodes and each node has 32 ranks,
        # if we set the partition world size to 32, then
        # replication_groups = [[0,32], [1,33], ..., [31,63]]
        replication_groups = []
        for i in range(0, partition_group_size):
            replication_groups.append(
                [j for j in range(i, i + world_size, partition_group_size)]
            )

        return MiCSShardingConfig(
            partition_groups=partition_groups,
            partition_rank=partition_rank,
            partition_group_size=partition_group_size,
            replication_groups=replication_groups,
            replication_group_size=replication_group_size,
        )

    # add sharding_groups, sharding_rank and sharding_world_size into fsdp params
    def init_fsdp_sharding_args(self, fsdp_params):
        # Initilize FSDP with partition groups only when there are more than one partition groups
        if not self.mics_enabled:
            return fsdp_params

        fsdp_sharding_params = dict(**fsdp_params)
        fsdp_sharding_params[
            "sharding_groups"
        ] = self.mics_sharding_cfg.partition_groups
        fsdp_sharding_params["sharding_rank"] = self.mics_sharding_cfg.partition_rank
        fsdp_sharding_params[
            "sharding_world_size"
        ] = self.mics_sharding_cfg.partition_group_size
        fsdp_sharding_params["auto_wrapper_callable"] = XlaFullyShardedDataParallelMiCS

        return fsdp_sharding_params

    def mics_gradient_sync(self):
        # All-reduce within replication groups only when replication world size is larger than one
        if self.mics_enabled:
            for m in self.modules():
                if not isinstance(m, XlaFullyShardedDataParallelMiCS):
                    continue
                # Each XlaFullyShardedDataParallelMiCS instance may have its own sharding config
                for p in [p for p in m.sharded_params if p.grad is not None]:
                    # Do div first to avoid overflow
                    p.grad.data.div_(1.0 * m.mics_sharding_cfg.replication_group_size)
                    xm.all_reduce(
                        xm.REDUCE_SUM,
                        [p.grad.data],
                        groups=m.mics_sharding_cfg.replication_groups,
                    )

    def clip_grad_norm_(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        groups: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        # override FSDP's clip_grad_norm_ by assigning MiCS'
        # partition_groups to the 'groups' parameter
        if groups is None:
            groups = self.mics_sharding_cfg.partition_groups
        return XlaFullyShardedDataParallel.clip_grad_norm_(
            self, max_norm=max_norm, norm_type=norm_type, groups=groups
        )

    @property
    def mics_enabled(self) -> bool:
        # MiCS is enabled when the replication world size is larger than 1
        # otherwise it has only one partation group just like FSDP
        return self.mics_sharding_cfg.replication_group_size > 1
