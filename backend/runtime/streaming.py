"""Epic 7: Streaming Runtime — layer-by-layer streaming via SSE.

Enables a live debugger: streams each layer's processing data
as an SSE event instead of waiting for the full inference.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
from typing import Any, AsyncGenerator

import torch

from .event_bus import bus, LAYER_PROCESSED


async def stream_inference(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prompt: str,
) -> AsyncGenerator[str, None]:
    """Yield SSE events for each layer as inference progresses."""

    yield _sse("inference.started", {"prompt": prompt, "num_tokens": input_ids.size(1)})
    queue: asyncio.Queue = asyncio.Queue()

    attention_maps: list[list[torch.Tensor]] = []
    mlp_activations: list[torch.Tensor] = []
    attn_hooks: list = []
    mlp_hooks: list = []

    def _make_attn_hook(li: int):
        def _hook(module, inputs, outputs):
            attn = outputs[-1].detach().cpu()
            while len(attention_maps) <= li:
                attention_maps.append([])
            attention_maps[li].append(attn)
            bus.emit(LAYER_PROCESSED, layer=li, component="attention")
            queue.put_nowait({
                "event": "layer.processed",
                "data": {
                    "layer": li,
                    "component": "attention",
                    "shape": list(attn.shape),
                    "timestamp": time.time(),
                },
            })
        return _hook

    def _make_mlp_hook(li: int):
        def _hook(module, inputs, outputs):
            h = module.act(inputs[0]).detach().cpu()
            while len(mlp_activations) <= li:
                mlp_activations.append(None)
            mlp_activations[li] = h
            bus.emit(LAYER_PROCESSED, layer=li, component="mlp")
            queue.put_nowait({
                "event": "layer.processed",
                "data": {
                    "layer": li,
                    "component": "mlp",
                    "shape": list(h.shape),
                    "timestamp": time.time(),
                },
            })
        return _hook

    for li in range(len(model.transformer.h)):
        block = model.transformer.h[li]
        ah = block.attn.register_forward_hook(_make_attn_hook(li))
        mh = block.mlp.register_forward_hook(_make_mlp_hook(li))
        attn_hooks.append(ah)
        mlp_hooks.append(mh)

    loop = asyncio.get_event_loop()

    def _run_inference():
        with torch.no_grad():
            model(input_ids, output_attentions=True, output_hidden_states=True)
        queue.put_nowait({"event": "inference.done", "data": {}})

    with concurrent.futures.ThreadPoolExecutor() as pool:
        task = loop.run_in_executor(pool, _run_inference)

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60.0)
                yield _sse(msg["event"], msg["data"])
                if msg["event"] == "inference.done":
                    break
            except asyncio.TimeoutError:
                yield _sse("inference.error", {"error": "inference timed out"})
                break

    for h in attn_hooks + mlp_hooks:
        h.remove()

    # Generation pass
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    yield _sse("inference.finished", {
        "generated_text": full_text,
        "num_layers": len(attention_maps),
        "num_attention_maps": sum(len(x) for x in attention_maps),
    })


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
