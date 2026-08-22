from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Any


def clip_sequence_limit(pipe: Any) -> int | None:
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return None
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    encoder = getattr(pipe, "text_encoder", None)
    encoder_limit = getattr(getattr(encoder, "config", None), "max_position_embeddings", None)
    limits: list[int] = []
    if encoder_limit is not None:
        limits.append(int(encoder_limit))
    if tokenizer_limit is not None:
        value = int(tokenizer_limit)
        if value <= 10_000:
            limits.append(value)
        elif encoder_limit is None:
            limits.append(value)
    return min(limits) if limits else None


def tokenize_for_chunking(tokenizer: Any, text: str) -> Any:
    old_limit = getattr(tokenizer, "model_max_length", None)
    try:
        if isinstance(old_limit, int) and old_limit <= 10_000:
            tokenizer.model_max_length = 100_000
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*sequence length is longer than the specified maximum.*",
                category=UserWarning,
            )
            return tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                verbose=False,
                return_tensors="pt",
            ).input_ids[0]
    finally:
        if isinstance(old_limit, int):
            tokenizer.model_max_length = old_limit


@contextmanager
def suppress_tokenizer_length_log():
    logger = logging.getLogger("transformers.tokenization_utils_base")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def shrink_ids_for_stable_decode(tokenizer: Any, token_ids: Any, capacity: int) -> Any:
    candidate = token_ids.detach().clone()
    while candidate.numel() > 0:
        decoded = tokenizer.decode(candidate, skip_special_tokens=True).strip()
        if decoded and tokenize_for_chunking(tokenizer, decoded).numel() <= capacity:
            return candidate
        candidate = candidate[:-1]
    return candidate


def sdxl_chunk_text(pipe: Any, tokenizer: Any, token_ids: Any, capacity: int) -> str:
    tokenizer_2 = getattr(pipe, "tokenizer_2", None)
    candidate = shrink_ids_for_stable_decode(tokenizer, token_ids, capacity)
    while candidate.numel() > 0:
        text = tokenizer.decode(candidate, skip_special_tokens=True).strip()
        if not text:
            return ""
        if tokenizer_2 is None:
            return text
        capacity_2 = max(1, int(getattr(tokenizer_2, "model_max_length", capacity)) - 2)
        if tokenize_for_chunking(tokenizer_2, text).numel() <= capacity_2:
            return text
        candidate = candidate[:-1]
    return ""


def average_sdxl_prompt_embeds(
    pipe: Any, text: str, clip_skip: int | None
) -> tuple[Any | None, Any | None]:
    tokenizer = pipe.tokenizer
    limit = clip_sequence_limit(pipe)
    if limit is None:
        return None, None
    capacity = max(1, limit - 2)
    if hasattr(pipe, "maybe_convert_prompt"):
        text = pipe.maybe_convert_prompt(text, tokenizer)
    token_ids = tokenize_for_chunking(tokenizer, text)
    if token_ids.numel() <= capacity:
        return None, None

    device = pipe._execution_device
    sequence_embeddings: list[Any] = []
    pooled_embeddings: list[Any] = []
    with suppress_tokenizer_length_log():
        for start in range(0, token_ids.numel(), capacity):
            chunk_text = sdxl_chunk_text(
                pipe, tokenizer, token_ids[start : start + capacity], capacity
            )
            if not chunk_text:
                continue
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=chunk_text,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=None,
                negative_prompt_2=None,
                clip_skip=clip_skip,
            )
            sequence_embeddings.append(prompt_embeds)
            pooled_embeddings.append(pooled_prompt_embeds)
    if not sequence_embeddings or not pooled_embeddings:
        return None, None

    torch = __import__("torch")
    return (
        torch.stack(sequence_embeddings, dim=0).mean(dim=0),
        torch.stack(pooled_embeddings, dim=0).mean(dim=0),
    )


def build_sdxl_prompt_kwargs(
    pipe: Any, prompt: str, negative_prompt: str, clip_skip: int
) -> dict[str, Any]:
    effective_clip_skip = clip_skip if clip_skip > 0 else None
    kwargs: dict[str, Any] = {"clip_skip": effective_clip_skip}
    if getattr(pipe, "tokenizer_2", None) is None:
        kwargs.update(prompt=prompt, negative_prompt=negative_prompt.strip() or None)
        return kwargs
    prompt_embeds, pooled_prompt_embeds = average_sdxl_prompt_embeds(
        pipe, prompt, effective_clip_skip
    )
    if prompt_embeds is None:
        kwargs["prompt"] = prompt
    else:
        kwargs.update(
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

    negative = negative_prompt.strip()
    negative_embeds, negative_pooled_embeds = average_sdxl_prompt_embeds(
        pipe, negative, effective_clip_skip
    ) if negative else (None, None)
    if negative_embeds is None:
        kwargs["negative_prompt"] = negative or None
    else:
        kwargs.update(
            negative_prompt=None,
            negative_prompt_embeds=negative_embeds,
            negative_pooled_prompt_embeds=negative_pooled_embeds,
        )
    return kwargs
