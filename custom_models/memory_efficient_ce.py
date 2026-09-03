"""Loss-only LM-head forwards for Qwen 3.5 training.

The stock Qwen 3.5 forward projects every token to the vocabulary before it
computes loss.  These forwards deliberately do the opposite: they give the
hidden states and LM-head weights directly to an efficient linear CE kernel.
"""

from types import MethodType

import torch


def _make_qwen3_5_loss_only_forward(linear_cross_entropy):
    # Importing here keeps normal installations usable without optional kernels.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None,
        pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
        video_grid_thw=None, mm_token_type_ids=None, logits_to_keep=0, **kwargs,
    ):
        if labels is None:
            return self._drope_original_forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=None, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep, **kwargs,
            )
        outputs = self.model(
            input_ids=input_ids, pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw, position_ids=position_ids,
            attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, mm_token_type_ids=mm_token_type_ids, **kwargs,
        )
        loss = linear_cross_entropy(outputs[0], self.lm_head.weight, labels)
        return Qwen3_5CausalLMOutputWithPast(
            loss=loss, logits=None, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    return forward


def apply_memory_efficient_ce(model, implementation: str):
    """Install one of ``baseline``, ``cce``, ``liger``, or ``quack``.

    Qwen 3.5 is supported by Liger's public model patch.  CCE and Quack expose
    linear-loss APIs, so they share the small Qwen-specific adapter above.
    """
    implementation = implementation.lower()
    if implementation == "baseline":
        return model
    if implementation == "liger":
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5

        apply_liger_kernel_to_qwen3_5(
            model=model, rope=False, rms_norm=False, swiglu=False,
            fused_linear_cross_entropy=True,
        )
        return model

    if not hasattr(model, "_drope_original_forward"):
        model._drope_original_forward = model.forward
    if implementation == "cce":
        from cut_cross_entropy import linear_cross_entropy

        def cce_loss(hidden_states, weight, labels):
            return linear_cross_entropy(hidden_states, weight, labels, shift=1)
        model.forward = MethodType(_make_qwen3_5_loss_only_forward(cce_loss), model)
    elif implementation == "quack":
        from quack.linear_cross_entropy import chunked_linear_cross_entropy

        def quack_loss(hidden_states, weight, labels):
            # Quack's TMA path needs a multiple of eight rows.  The causal
            # shift leaves 32767 rows for this run, so append one ignored row.
            x = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
            target = labels[:, 1:].reshape(-1)
            padding = (-x.shape[0]) % 8
            if padding:
                x = torch.cat((x, x.new_zeros((padding, x.shape[-1]))))
                target = torch.cat((target, target.new_full((padding,), -100)))
            return chunked_linear_cross_entropy(x, weight, target, chunk_size=4096)
        model.forward = MethodType(_make_qwen3_5_loss_only_forward(quack_loss), model)
    else:
        raise ValueError(
            "memory_efficient_ce must be baseline, cce, liger, or quack; "
            f"got {implementation!r}"
        )
    return model
