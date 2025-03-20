import os
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from accelerate.logging import get_logger
from accelerate.utils.constants import FSDP_MODEL_NAME, OPTIMIZER_NAME
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed.checkpoint as dcp

logger = get_logger(__name__)

# - variable to capture the model variable
#   in the save/load model calls
MODEL_INDEX = None
KEY_MODEL = "model"
KEY_OPTIMIZER = "optimizer"

# Below are rewrite of HF FSDP model saving functions to be able to handle
# that the parameters are now a mixture of regular and Dtensors.
# - these functions are found in accelerate.utils.fsdp_utils.py
# - save_fsdp_model, save_fsdp_optimizer, load_fsdp_model, load_fsdp_optimizer
# NOTE: we will observe warnings such as
# /torch/distributed/checkpoint/state_dict.py:520:
# FutureWarning: Please use DTensor instead and we are deprecating ShardedTensor.


# rewrite of func from accelerate.utils.fsdp_utils.py
# - empty function, the main logic will be in save_fsdp_optimizer (see below).
def save_fsdp_model(
    fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False
):
    # pylint: disable=global-statement
    global MODEL_INDEX
    MODEL_INDEX = model_index


# rewrite of func from accelerate.utils.fsdp_utils.py
# - saves both model and optimizer at the same time
def save_fsdp_optimizer(
    fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0
):

    if fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
        raise NotImplementedError(
            "Checkpointing for FSDP2 only enabled for sharded state dict."
        )

    # get the state dicts for model and optimize
    (model_state_dict, optimizer_state_dict) = get_state_dict(model, optimizer)

    # - save model
    ckpt_model = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{MODEL_INDEX}")
    os.makedirs(ckpt_model, exist_ok=True)
    logger.info(f"Saving model to {ckpt_model}")
    dcp.save(
        state_dict={KEY_MODEL: model_state_dict},
        storage_writer=dcp.FileSystemWriter(ckpt_model),
        planner=DefaultSavePlanner(),
    )
    logger.info(f"Model saved to {ckpt_model}")

    # - save optimizer
    ckpt_opt = os.path.join(output_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
    os.makedirs(ckpt_opt, exist_ok=True)
    logger.info(f"Saving Optimizer state to {ckpt_opt}")
    dcp.save(
        state_dict={KEY_OPTIMIZER: optimizer_state_dict},
        storage_writer=dcp.FileSystemWriter(ckpt_opt),
        planner=DefaultSavePlanner(),
    )
    logger.info(f"Optimizer state saved in {ckpt_opt}")


# rewrite of func from accelerate.utils.fsdp_utils.py
# - empty function, main logic in load_fsdp_optimizer (see below).
def load_fsdp_model(
    fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False
):
    # pylint: disable=global-statement
    global MODEL_INDEX
    MODEL_INDEX = model_index


# rewrite of func from accelerate.utils.fsdp_utils.py
# - loads both model and optimizer
def load_fsdp_optimizer(
    fsdp_plugin,
    accelerator,
    optimizer,
    model,
    input_dir,
    optimizer_index=0,
    adapter_only=False,
):

    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
        raise NotImplementedError(
            "Checkpointing for FPSD2 only enabled for sharded state dict."
        )

    # - get the state dicts
    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)

    # - load the model state dict
    ckpt_model = os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{MODEL_INDEX}")
    dcp.load(
        state_dict={KEY_MODEL: model_state_dict},
        storage_reader=dcp.FileSystemReader(ckpt_model),
        planner=DefaultLoadPlanner(),
    )

    # - load the optimizer state dict
    ckpt_opt = os.path.join(input_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
    dcp.load(
        state_dict={KEY_OPTIMIZER: optimizer_state_dict},
        storage_reader=dcp.FileSystemReader(ckpt_opt),
        planner=DefaultLoadPlanner(),
    )

    # - set the state dicts
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state_dict,
        optim_state_dict=optimizer_state_dict,
    )

    # FIXME:
    # - We see errors that occur in optimizer.step()
    # - torch/optim/optimizer.py", line 89, in _use_grad
    # - torch/optim/adamw.py", line 214, in step beta1,
    #   beta2 = cast(Tuple[float, float], group["betas"])
    # - KeyError: 'betas'
    # - Fortunately, this seems to be limited to the empty groups case, where
    #   it seems that it is just the params are not initialized. Since we suppose
    #   these groups are never used, we simply initialize the empty groups with
    #   random values so the errors do not throw.
    for group in optimizer.param_groups:
        if len(group["params"]) == 0:
            group["betas"] = (0.9, 0.999)
            group["lr"] = 0.0
            group["initial_lr"] = 0.0
            group["eps"] = 1e-8
            group["weight_decay"] = 0.0

