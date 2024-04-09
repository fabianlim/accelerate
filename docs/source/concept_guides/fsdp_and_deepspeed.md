<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Moving between FSDP And DeepSpeed [DRAFT]

🤗 Accelerate offers flexibilty of training frameworks, by integrating two extremely powerful tools for distributed training, namely [Pytorch FSDP](../usage_guides/fsdp.md) and [Microsoft DeepSpeed](../usage_guides/deepspeed.md). The aim fo this article is to draw parallels, as well as to outline potential differences, to empower the user to switch seamlessly between these two frameworks.

<Tip>
  To switch between the frameworks, we recommend 🤗 `accelerate launch`.

  Simply pass [FSDP and DeepSpeed arguments](../package_reference/cli#accelerate-launch) directly to `accelerate launch` or pass `--config_file`. No need for any code instrumentation!

  Exemplar 🤗 Accelerate configurations can be found here for [DeepSpeed](../usage_guides/deepspeed#accelerate-deepspeed-plugin) and [FSDP](../usage_guides/fsdp#how-it-works-out-of-the-box). 
 
</Tip>

<!--
The aim of this concept guide is to elucidate similarities/differences with empirical observations and code aspects.  We assume that 🤗 Accelerate is used and configured using the [FSDP and DeepSpeed arguments](../package_reference/cli.md#accelerate-launch). No TPU aspects are discussed. Also we focus only on single-node aspects.
-->

This article is written with 🤗 Accelerate in mind, for single-node, multi-GPU, scenarios only. No TPU aspects will be considered.

## Configuring Various Functionalities

Model tensors are split into different GPUs in attempt scale up model sizes; this is termed *sharding* in FSDP, and *partitioning* in DeepSpeed. The FSDP sharding strategy and the DeepSpeed ZeRO Stages, are configured by `--fsdp_sharding_strategy`, and `--zero_stage`, respectively.  For example FSDP `FULL_SHARD` maps to DeepSpeed ZeRO stage `3`, see this for a comprehensive [mapping between FSDP sharding and DeepSpeed ZeRO settings](../usage_guides/fsdp#mapping-between-fsdp-sharding-strategies-and-deepspeed-zero-stages). There exists a host of other settings besides these two, summarized by similarity in the below table.

Group | Framework | Configuration | Example | Restrictions (if any)
--|--|--|--|--
sharding / partitioning | FSDP<br>DeepSpeed | `--fsdp_sharding_strategy`<br>`--zero_stage` | `1` (`FULL_SHARD`) <br>`3` | 
offload | FSDP<br>DeepSpeed | `--fsdp_offload_params`<br>`--offload_param_device`<br>`--offload_optimizer_device` | `true`<br>`cpu`<br>`cpu` | all or nothing <br><br> 
model loading | FSDP<br>DeepSpeed | <span style="white-space:nowrap;">`--fsdp_cpu_ram_efficient_loading`</span><br>`--zero3_init_flag` | `true`<br>`true` | <br>only ZeRO 3
efficient checkpointing | FSDP<br>DeepSpeed | `--fsdp_state_dict_type`<br>`--zero3_save_16bit_model` |  `SHARDED_STATE_DICT`<br>`true` |  <br>only ZeRO 3
pipeline | FSDP<br><br>DeepSpeed | `--fsdp_forward_prefetch`<br>`--fsdp_backward_prefetch`<br>None | `true`<br>`BACKWARD_PRE` | <br><br>?? check for DS
model | FSDP<br><br>DeepSpeed |  `--fsdp_auto_wrap_policy`<br><span style="white-space:nowrap;">`--fsdp_transformer_layer_cls_to_wrap`</span><br>None | `TRANSFORMER_BASED_WRAP`<br><Layer Class> |<br>Usually not needed <br>Transparent to user.
parameters summoning | FSDP<br>DeepSpeed | `--fsdp_use_orig_params`<br>None | `true` | required for `torch.compile`<br>Transparent to user
parameters syncing | FSDP<br>DeepSpeed | `--fsdp_sync_module_states`<br>None | `true` | <br>?? Need to check
training | FSDP<br>DeepSpeed | None<br>`--gradient_accumulation_steps`<br>`--gradient_clipping` | <br>`auto`<br>`auto` | Transparent to user

We reiterate again that for all possible settings for the above, refer to [`Accelerate` launch documentation](../package_reference/cli#accelerate-launch).

<Tip>

    To access other DeepSpeed configurations, such as mixed precision settings, 
    one has to link a `deepspeed_config_file`, see [instructions here](../usage_guides/deepspeed#deepspeed-config-file).  
    
</Tip>

<!--
Since the terminologies used in FSDP and DeepSpeed are disparate, there are various guides that [reconcile the mapping between FSDP sharding and DeepSpeed ZeRO](../usage_guides/fsdp#mapping-between-fsdp-sharding-strategies-and-deepspeed-zero-stages). This section aims to reconcile these differences for a 🤗 Accelerate user that desires to use both frameworks in an equivalent manner.
In general, DeepSpeed offers more fine-grained control, but may incur extra memory consumption as model and optimizer parameters are always upcasted to `float32`. 
-->

<Tip>
    When using FSDP these are recommendations that should be followed.

    - set `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`. Do not set `fsdp_transformer_layer_cls_to_wrap` if using the latest `transformers` versions.
    - if `fsdp_cpu_ram_efficient_loading: True`, set `fsdp_sync_module_states: True` otherwise the model will not load properly.
    - when using `torch.compile` set `fsdp_use_orig_params: True`.

</Tip>

<Tip>
    When using DeepSpeed these are recommendations that should be followed.

    - set `gradient_accumulation_steps: "auto"` and `gradient_clipping: "auto"` to automatically pick up values set in [`TrainingArguments`].

</Tip>

TODO: Consider elaborating on some points ? Maybe a small subsection for each?
- on checkpoint sharding (DS does not have checkpoint sharding?) Des FSDP save the model in 16 bit also?
- zero3 init in DS will be automatic if model is above a certain size. More details on this?
- do I need to talk about bucketing in DS? this is how DS takes care of partitioning in an automatic manner without the wrap policy. Do I need to explain FSDP wrapping causes graph breaks in torch.compile and how this is being resolved?
- how does DS take care of pipelining? I need to check further.
- do we need to discuss parameter summoning for DS? maybe not because this is an advanced usage
- should we discusss activation checkpointing in the configs or is this obvious?


## On Data Precision

To discuss the how data precision is handled in both FSDP and Deepspeed, it is instructive to first give a flow of how model parameters are handled in these frameworks. Before the model / optimizer parameters are distributed across GPUs, parameter preparation is involved to first "flatten" them to  one-dimensional [`torch.Tensor`]'s. The implementation of FSDP / DeepSpeed varies in the respect of the `dtype` in which these "flattened" parameters are stored, and there are ramnifications with regards to how [`torch.Optimizer`]'s allocate their `dtypes`'s. The table below outlines the processes for both frameworks; the "Local" column indicates the process occurring at a per-gpu level, therefore any memory overheads by upcasting should be understood to be amortized by the number of gpus used.

<!--
TODO: for FSDP there are some mixed precision settings like `keep_low_precision_grads`, should we discuss them? NVM, because they way huggingface prepares the model, keep_low_precision_grads will never need to be used
-->

Process | Local | Framework | Details
--|--|--|--
Loading, i.e., [`AutoModel.from_pretrained(..., torch_dtype)`] |  
Preparation, i.e., creation of "flat params" | ✅ | FSDP<br>DeepSpeed | created in `torch_dtype`.<br> disregards `torch_dtype`, created in `float32`.
Optimizer initialization | ✅ | FSDP<br>DeepSpeed  | creates parameters in `torch_dtype`<br> creates parameters in `float32`
Training Step, i.e, forward, backward, reduction | | FSDP<br>DeepSpeed  | follows [`MixedPrecision`](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision)<br> follows `deepspeed_config_file` mixed precision settings.
Optimizer (Pre-Step) | ✅ | FSDP<br>DeepSpeed | upcasting (if any) to `torch_dtype`<br>upcasted to `float32`
Optimizer (Actual Step) | ✅ | FSDP<br>DeepSpeed  | occurs in `torch_dtype` <br> occurs in `float32`.


<!--
Both FSDP and DeepSpeed have logic for sharding/partitioning the model / optimizer parameters across GPUs; however there are some differences to be aware of. Both of these
frameworks will create new [`torch.Tensor`]'s to hold "flattened" parameters, which will be used to instantiate the optimizers in each shard/partition. However since the [`torch.Optimizer`]'s allocate their `dtypes`'s based on the optimized parameters, it is important to note the following:

For FSDP there are two options i) load and train/optimize the model in low precision, or ii) load in full precision and configure [`MixedPrecision`](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision) to have activations / reduction performed in low precision. But for DeepSpeed always load the model in full precision. To summarize:
-->

<Tip warning={true}>

    Therefore when using DeepSpeed, one should always load the model with `torch_dtype=torch.float32`.

    However if the number of GPUs are small, be aware of potentially significant memory overheads due to the "local" upcasting.

</Tip>

<Tip warning={true}>

    With FSDP it is possible to operate the [`torch.Optimizer`] in low precision `torch_dtype`, which may be helpful when using small number of GPUs.

    And if migrating to DeepSpeed from FSDP, be aware that if `torch_dtype` had been previously set to low precision (unnecessarily, see above), it will result in non-equivalent observations since then FSDP will optimize in low precision.

</Tip>


To clarify the above table consider the concrete examples below; the optimizer pre- and actual step combined for brevity. Thus to ensure that the same data precisions are used in both FSDP and DeepSpeed, always specify `torch_dtype=torch.float32` while calling [`AutoModel.from_pretrained`].

Framework | Model Loading (`torch_dtype`) | Mixed Precision | Preparation (Local) | Training | Optimizer (Local)
--|--|--|--|--|--
FSDP | bf16 | default (none) | bf16 | bf16 | bf16
FSDP | fp32 | bf16 | fp32 | bf16 | fp32
DeepSpeed   | fp32 | bf16 | fp32 | bf16 | fp32