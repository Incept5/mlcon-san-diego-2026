#!/usr/bin/env python3
import os
import sys
import numpy as np
import gradio as gr
from llama_cpp import Llama
import plotly.graph_objects as go
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

os.environ.setdefault("GGML_METAL_LOG_LEVEL", "1")

DEFAULT_MODEL_PATH = "/Users/jdavies/.lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf"
DEFAULT_GPU_LAYERS = -1
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_SYSTEM_PROMPT = "You are a Yoda assistant, you always reply like Yoda and often mention the force. You are wise with your words."
DEFAULT_USER_PROMPT = "Where is Westminster Abbey?"
DEFAULT_GENERATE_PROMPT = "The cat shat on the"

# Fixed display size: top 5 candidate tokens + an "other" wedge for the remainder.
NUM_TOKENS_DISPLAYED = 5

# Qwen3-series recommended sampling for *non-thinking* mode (the mode this app forces).
# Source: Qwen team guidance for Qwen3 chat models; presence/repetition penalties off by default.
QWEN3_NO_THINK_DEFAULTS = dict(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repeat_penalty=1.0,
    num_tokens=NUM_TOKENS_DISPLAYED,
)


@dataclass
class InferenceConfig:
    temperature: float = QWEN3_NO_THINK_DEFAULTS["temperature"]
    top_p: float = QWEN3_NO_THINK_DEFAULTS["top_p"]
    top_k: int = QWEN3_NO_THINK_DEFAULTS["top_k"]
    repeat_penalty: float = QWEN3_NO_THINK_DEFAULTS["repeat_penalty"]
    num_tokens: int = QWEN3_NO_THINK_DEFAULTS["num_tokens"]


class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        self._devnull = open(os.devnull, "w")
        sys.stderr = self._devnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._original_stderr
        self._devnull.close()


def build_chat_prompt(system_prompt: str, user_prompt: str, assistant_prefix: str = "") -> str:
    """Qwen3 ChatML prompt with thinking mode forcibly disabled.

    Two complementary mechanisms (both required for reliable no-think behavior):
      1. `/no_think` directive appended to the system prompt.
      2. Pre-seeded empty <think>\\n\\n</think>\\n\\n block at the start of the assistant turn,
         so the very first token the model emits is the actual answer.
    """
    system_text = system_prompt.strip() if system_prompt.strip() else "You are a helpful assistant."
    return (
        f"<|im_start|>system\n{system_text} /no_think<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_prefix}"
    )


class TokenProbabilityAnalyzer:
    # Special tokens we treat as "stop signals". We probe the model's vocab for
    # any that resolve to a single token id and report their probability at
    # every step — even when they fall outside the top-5 display set.
    STOP_TOKEN_LITERALS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>", "</s>")

    def __init__(self, model_path: str, n_ctx: int = DEFAULT_CONTEXT_SIZE, n_gpu_layers: int = 0):
        with SuppressStderr():
            self.model = Llama(
                model_path=model_path,
                logits_all=True,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                chat_format="chatml",
            )
        self.debug_info = ""
        self.last_prompt = ""
        self.stop_token_ids = self._resolve_stop_token_ids()
        self.last_stop_probs: List[Tuple[str, int, float]] = []  # [(literal, token_id, prob)]

    def _resolve_stop_token_ids(self) -> List[Tuple[str, int]]:
        resolved: List[Tuple[str, int]] = []
        for literal in self.STOP_TOKEN_LITERALS:
            try:
                ids = self.model.tokenize(literal.encode("utf-8"), add_bos=False, special=True)
            except Exception:
                continue
            if len(ids) == 1:
                resolved.append((literal, ids[0]))
        return resolved

    def analyze_prompt(self, prompt: str, config: InferenceConfig) -> List[Tuple[str, float, float]]:
        """Return (token_str, probability, raw_logit) for the most likely next tokens.

        Uses the low-level eval/scores API. This is *far* more reliable than
        create_completion(logprobs=N), which intermittently returns empty
        top_logprobs on finish_reason='length' — the bug the old retry hack tried to paper over.
        """
        self.last_prompt = prompt
        token_ids = self.model.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        self.debug_info = f"Prompt: {len(prompt)} chars → {len(token_ids)} tokens\n"

        n_ctx = self.model.n_ctx()
        if len(token_ids) >= n_ctx:
            self.debug_info += f"ERROR: prompt ({len(token_ids)} tokens) exceeds context window ({n_ctx})\n"
            return []

        self.model.reset()
        self.model.eval(token_ids)

        # scores[i] are the logits predicting position i+1; the next-token logits
        # therefore live at scores[n_tokens - 1].
        last_idx = self.model.n_tokens - 1
        logits = np.array(self.model.scores[last_idx], dtype=np.float64, copy=True)

        self._apply_repeat_penalty(logits, token_ids, config.repeat_penalty)
        return self._top_tokens_from_logits(logits, config)

    @staticmethod
    def _apply_repeat_penalty(logits: np.ndarray, token_ids: List[int], penalty: float) -> None:
        if penalty == 1.0 or not token_ids:
            return
        for tid in set(token_ids):
            if 0 <= tid < logits.shape[0]:
                logits[tid] = logits[tid] / penalty if logits[tid] > 0 else logits[tid] * penalty

    def _top_tokens_from_logits(self, logits: np.ndarray, config: InferenceConfig) -> List[Tuple[str, float, float]]:
        # Temperature-scaled softmax, numerically stable.
        temp = max(config.temperature, 1e-6)
        scaled = (logits - logits.max()) / temp
        exp = np.exp(scaled)
        probs = exp / exp.sum()

        # Capture stop-token probabilities BEFORE filtering, so the UI can show
        # them even when EOS is ranked far below the top 5 (typical mid-paragraph
        # in Generate mode — exactly the "why doesn't it stop?" moment).
        self.last_stop_probs = [
            (literal, tid, float(probs[tid]))
            for literal, tid in self.stop_token_ids
            if 0 <= tid < probs.shape[0]
        ]

        # Always surface the top `num_tokens` candidates by *raw* probability.
        # We deliberately do NOT use top_p/top_k as display filters: with an
        # aggressive top_p (e.g. 0.8 when the model is confident) the survivor
        # set collapses to a single token, hiding the long tail this tool exists
        # to show. top_p/top_k remain in the UI as sampler-reference values.
        n = min(config.num_tokens, len(probs))
        if n <= 0:
            return []
        top_idx = np.argpartition(-probs, n - 1)[:n]
        top_idx = top_idx[np.argsort(-probs[top_idx])]

        results: List[Tuple[str, float, float]] = []
        for idx in top_idx:
            idx_int = int(idx)
            try:
                token_str = self.model.detokenize([idx_int]).decode("utf-8", errors="replace")
            except Exception:
                token_str = f"<id:{idx_int}>"
            # Absolute probability over the full vocab — no renormalization.
            results.append((token_str, float(probs[idx]), float(logits[idx])))
        return results

    def stop_rank_for(self, token_id: int) -> Optional[int]:
        """Return the 1-based rank of a token id in the most recent distribution,
        if we still have the logits accessible. Used only for the readout."""
        try:
            last_idx = self.model.n_tokens - 1
            logits = np.array(self.model.scores[last_idx], dtype=np.float64, copy=False)
            # 1-based rank: how many tokens have a higher logit
            return int((logits > logits[token_id]).sum()) + 1
        except Exception:
            return None


current_analyzer: Optional[TokenProbabilityAnalyzer] = None
last_analysis_results: Optional[List[Tuple[str, float, float]]] = None


def load_model(model_path: str, context_size: int, gpu_layers: int) -> str:
    global current_analyzer, last_analysis_results
    if not model_path.strip():
        return "Error: Model path cannot be empty"
    try:
        current_analyzer = TokenProbabilityAnalyzer(
            model_path=model_path, n_ctx=int(context_size), n_gpu_layers=int(gpu_layers)
        )
        last_analysis_results = None
        return f"✅ Model loaded: {Path(model_path).name}"
    except Exception as e:
        return f"❌ Error: {e}"


STOP_TOKENS = {"</s>", "<|endoftext|>", "<|im_end|>", "<|end|>", "<|eot_id|>", "<eos>"}
WHITESPACE_CONTROLS = {"\n", "\t", "\r"}


def is_stop_token(token: str) -> bool:
    if token in STOP_TOKENS:
        return True
    if not token:
        return True
    # Single-char unprintable control (excluding ordinary whitespace) counts as a stop;
    # newlines and tabs are normal generation tokens and must not disable the flow.
    if len(token) == 1 and ord(token) < 32 and token not in WHITESPACE_CONTROLS:
        return True
    return False


def _disabled_buttons():
    """Default 5-button state shown before any analysis exists."""
    return [gr.Button(value=f"#{i + 1}", interactive=False) for i in range(NUM_TOKENS_DISPLAYED)]


def _button_updates_for(token_probs):
    """Label each candidate button with its rank + token preview, and disable
    buttons whose token is a stop token (clicking them would terminate)."""
    updates = []
    for i in range(NUM_TOKENS_DISPLAYED):
        if i < len(token_probs):
            token, _prob, _logit = token_probs[i]
            preview = repr(token)
            if len(preview) > 14:
                preview = preview[:11] + "…"
            label = f"#{i + 1} {preview}"
            updates.append(gr.Button(value=label, interactive=not is_stop_token(token)))
        else:
            updates.append(gr.Button(value=f"#{i + 1}", interactive=False))
    return updates


def _analyze_response(token_probs, formatted_prompt_text):
    if not token_probs:
        msg = (
            "**No tokens returned.**\n\n"
            f"```\n{current_analyzer.debug_info if current_analyzer else ''}```"
        )
        return (msg, None, *_disabled_buttons(), formatted_prompt_text)
    stop_probs = list(current_analyzer.last_stop_probs) if current_analyzer else []
    body = format_top_tokens(token_probs, stop_probs)
    chart = create_pie_chart(token_probs)
    return (body, chart, *_button_updates_for(token_probs), formatted_prompt_text)


def analyze_tokens(mode, system_prompt, user_prompt, generate_prompt,
                   temperature, top_p, top_k, repeat_penalty):
    global current_analyzer, last_analysis_results

    if current_analyzer is None:
        return ("⚠️ No model loaded — open the **Model Configuration** tab to load one.",
                None, *_disabled_buttons(), "")

    active_prompt = user_prompt if mode == "Chat" else generate_prompt
    if not active_prompt or not active_prompt.strip():
        return ("⚠️ Prompt is empty — type something before clicking Analyze.",
                None, *_disabled_buttons(), "")

    config = InferenceConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        num_tokens=NUM_TOKENS_DISPLAYED,
    )

    try:
        prompt = build_chat_prompt(system_prompt, user_prompt) if mode == "Chat" else generate_prompt
        token_probs = current_analyzer.analyze_prompt(prompt, config)
        last_analysis_results = token_probs
        return _analyze_response(token_probs, current_analyzer.last_prompt)
    except Exception as e:
        import traceback
        return (f"Error: {e}\n\n```\n{traceback.format_exc()}```",
                None, *_disabled_buttons(), "")


def add_token_by_rank(rank, mode, system_prompt, user_prompt, generate_prompt, chat_response,
                      temperature, top_p, top_k, repeat_penalty):
    """Inject the candidate at 1-based `rank` (1..NUM_TOKENS_DISPLAYED) and re-analyze."""
    global current_analyzer, last_analysis_results

    if not last_analysis_results or rank < 1 or rank > len(last_analysis_results):
        return ("Run **Analyze** first.", None, *_disabled_buttons(),
                generate_prompt, chat_response, "")

    chosen_token = last_analysis_results[rank - 1][0]
    config = InferenceConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        num_tokens=NUM_TOKENS_DISPLAYED,
    )

    try:
        if mode == "Chat":
            new_chat_response = chat_response + chosen_token
            prompt = build_chat_prompt(system_prompt, user_prompt, assistant_prefix=new_chat_response)
            new_generate_prompt = generate_prompt
        else:
            new_generate_prompt = generate_prompt + chosen_token
            prompt = new_generate_prompt
            new_chat_response = chat_response

        token_probs = current_analyzer.analyze_prompt(prompt, config)
        last_analysis_results = token_probs
        body, chart, *btn_and_prompt = _analyze_response(token_probs, current_analyzer.last_prompt)
        *btn_updates, prompt_view = btn_and_prompt
        return (body, chart, *btn_updates, new_generate_prompt, new_chat_response, prompt_view)
    except Exception as e:
        import traceback
        return (f"Error: {e}\n\n```\n{traceback.format_exc()}```", None,
                *_disabled_buttons(), generate_prompt, chat_response, "")


def reset_state(mode):
    """Clear chat response / generate prompt back to defaults and wipe analysis state."""
    global last_analysis_results
    last_analysis_results = None
    if mode == "Chat":
        return (gr.update(value=""), gr.update(value=DEFAULT_USER_PROMPT),
                "Cleared. Click **Analyze** to start again.",
                None, *_disabled_buttons(), "")
    return (gr.update(value=""), gr.update(value=DEFAULT_GENERATE_PROMPT),
            "Cleared. Click **Analyze** to start again.",
            None, *_disabled_buttons(), "")


def _format_pct(prob: float) -> str:
    pct = prob * 100
    return f"{pct:.4f}%" if pct >= 0.0001 else f"{pct:.2e}%"


def format_top_tokens(token_probs: List[Tuple[str, float, float]],
                      stop_probs: Optional[List[Tuple[str, int, float]]] = None) -> str:
    lines = [f"**Top {len(token_probs)} candidate tokens**", ""]
    lines.append("| # | Token | Probability | Logit |")
    lines.append("|---:|---|---:|---:|")
    for i, (token, prob, logit) in enumerate(token_probs, 1):
        marker = "🛑 " if is_stop_token(token) else ""
        lines.append(f"| {i} | {marker}`{repr(token)}` | {prob * 100:.2f}% | {logit:.3f} |")
    # Combined stop-token row directly under the top-5, even when EOS is ranked
    # far below — the canonical "why doesn't it stop?" readout.
    if stop_probs:
        cum = sum(p for _, _, p in stop_probs)
        lines.append(f"| 🛑 | _stop tokens (combined)_ | {_format_pct(cum)} | — |")
    total = sum(p for _, p, _ in token_probs) * 100
    lines.append("")
    lines.append(f"_Top {len(token_probs)} cover **{total:.2f}%** of the full distribution._")
    return "\n".join(lines)


def create_pie_chart(token_probs: List[Tuple[str, float, float]]):
    if not token_probs:
        return None

    labels, values, colors, hover = [], [], [], []
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899",
               "#06b6d4", "#84cc16", "#f43f5e", "#a855f7", "#14b8a6"]
    palette_idx = 0
    for token, prob, logit in token_probs:
        display = repr(token)
        if len(display) > 22:
            display = display[:19] + "…"
        pct = prob * 100
        if is_stop_token(token):
            labels.append(f"🛑 {display} ({pct:.2f}%)")
            colors.append("#ef4444")
        else:
            labels.append(f"{display} ({pct:.2f}%)")
            colors.append(palette[palette_idx % len(palette)])
            palette_idx += 1
        values.append(prob)
        hover.append(f"token: {repr(token)}<br>probability: {pct:.3f}%<br>logit: {logit:.3f}")

    # The displayed tokens rarely cover 100% of the vocab — represent the rest
    # honestly with a grey "other" wedge so the pie shows the true distribution.
    shown = sum(values)
    if shown < 0.9995:
        remaining = 1.0 - shown
        labels.append(f"(other tokens — {remaining * 100:.2f}%)")
        values.append(remaining)
        colors.append("#e5e7eb")
        hover.append(f"all other vocab tokens<br>cumulative: {remaining * 100:.3f}%")

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, textinfo="none",
        marker=dict(colors=colors),
        hovertext=hover, hoverinfo="text",
        sort=False, direction="clockwise",
    )])
    fig.update_layout(
        title="Next-token probability distribution",
        height=310, font=dict(size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(font=dict(size=11)),
    )
    return fig


def create_gradio_interface():
    with gr.Blocks(title="Token Probability Analyzer") as app:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# 🔍 Token Probability Analyzer")
                gr.Markdown(
                    "Inspect what the model **would** sample next, token by token. "
                    "Qwen3 thinking mode is forcibly **off** "
                    "(`/no_think` directive + pre-seeded empty `<think></think>` block)."
                )
            with gr.Column(scale=2):
                mode = gr.Radio(
                    choices=["Chat", "Generate"], value="Chat", label="Mode",
                    info="Chat: separate response window  •  Generate: continue raw text",
                )
                model_status = gr.Markdown("_No model loaded_")

        with gr.Tab("Token Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    system_prompt = gr.Textbox(label="System Prompt (optional)", value=DEFAULT_SYSTEM_PROMPT, lines=2,
                                               placeholder="System instructions…", visible=True)
                    user_prompt = gr.Textbox(label="User Prompt", value=DEFAULT_USER_PROMPT, lines=2, visible=True)
                    chat_response = gr.Textbox(label="Assistant Response", value="", lines=4,
                                               placeholder="Response will grow here as you add tokens…", visible=True)
                    generate_prompt = gr.Textbox(label="Text to Continue", value=DEFAULT_GENERATE_PROMPT, lines=6,
                                                 placeholder="Text will grow here as you add tokens…", visible=False)

                    with gr.Accordion("Sampling parameters (Qwen3 no-think defaults)", open=False):
                        with gr.Row():
                            temperature = gr.Slider(0.01, 2.0, value=QWEN3_NO_THINK_DEFAULTS["temperature"],
                                                    label="Temperature")
                            top_p = gr.Slider(0.01, 1.0, value=QWEN3_NO_THINK_DEFAULTS["top_p"], label="Top-p")
                        with gr.Row():
                            top_k = gr.Slider(1, 100, value=QWEN3_NO_THINK_DEFAULTS["top_k"], step=1, label="Top-k")
                            repeat_penalty = gr.Slider(0.5, 2.0, value=QWEN3_NO_THINK_DEFAULTS["repeat_penalty"],
                                                       step=0.01, label="Repeat Penalty",
                                                       info="Qwen team recommends leaving this at 1.0.")

                    with gr.Row():
                        analyze_btn = gr.Button("🚀 Analyze", variant="primary")
                        reset_btn = gr.Button("🧹 Clear", variant="stop")

                    gr.Markdown("**Inject a candidate** (greedy = `#1`; pick any other to explore the path-not-taken):")
                    with gr.Row():
                        add_buttons = [
                            gr.Button(value=f"#{i + 1}", interactive=False,
                                      variant=("primary" if i == 0 else "secondary"))
                            for i in range(NUM_TOKENS_DISPLAYED)
                        ]

                    formatted_prompt_view = gr.Textbox(
                        label="Formatted prompt sent to model", lines=8, interactive=False,
                    )

                with gr.Column(scale=1):
                    chart = gr.Plot(label="")
                    results = gr.Markdown("Click **Analyze** to see results…")

        with gr.Tab("Model Configuration"):
            with gr.Row():
                with gr.Column():
                    model_path = gr.Textbox(
                        label="Model Path",
                        value=os.environ.get("GGUF_MODEL_PATH", DEFAULT_MODEL_PATH),
                        placeholder="Path to .gguf file",
                    )
                    with gr.Row():
                        context_size = gr.Number(label="Context Size", value=DEFAULT_CONTEXT_SIZE, precision=0)
                        gpu_layers = gr.Number(label="GPU Layers", value=DEFAULT_GPU_LAYERS, precision=0)
                    load_btn = gr.Button("🔄 Load Model", variant="primary")
                with gr.Column():
                    load_status = gr.Textbox(label="Status", lines=4, interactive=False)

        def toggle_mode(m, current_user, current_generate):
            # Preserve whatever text the user has typed; only seed the mode's
            # default prompt when the target field is empty (Gradio doesn't
            # always re-render the initial `value` of a component that was
            # created with visible=False).
            user_val = current_user if (current_user and current_user.strip()) else DEFAULT_USER_PROMPT
            gen_val = current_generate if (current_generate and current_generate.strip()) else DEFAULT_GENERATE_PROMPT
            if m == "Chat":
                return (gr.update(visible=True),
                        gr.update(visible=True, value=user_val),
                        gr.update(visible=True),
                        gr.update(visible=False))
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True, value=gen_val))

        # queue=False + show_progress="hidden": this is a pure UI toggle, never
        # touching the model, so it shouldn't sit in the queue or show a spinner.
        mode.change(fn=toggle_mode, inputs=[mode, user_prompt, generate_prompt],
                    outputs=[system_prompt, user_prompt, chat_response, generate_prompt],
                    queue=False, show_progress="hidden")

        def load_and_update(p, c, g):
            msg = load_model(p, c, g)
            return msg, msg
        load_btn.click(fn=load_and_update, inputs=[model_path, context_size, gpu_layers],
                       outputs=[load_status, model_status])

        analyze_btn.click(
            fn=analyze_tokens,
            inputs=[mode, system_prompt, user_prompt, generate_prompt,
                    temperature, top_p, top_k, repeat_penalty],
            outputs=[results, chart, *add_buttons, formatted_prompt_view],
        )

        # One click handler per candidate button; rank is bound at definition time
        # so each lambda captures its own value rather than the loop variable.
        for i, btn in enumerate(add_buttons):
            btn.click(
                fn=lambda *args, rank=i + 1: add_token_by_rank(rank, *args),
                inputs=[mode, system_prompt, user_prompt, generate_prompt, chat_response,
                        temperature, top_p, top_k, repeat_penalty],
                outputs=[results, chart, *add_buttons,
                         generate_prompt, chat_response, formatted_prompt_view],
            )

        reset_btn.click(
            fn=reset_state,
            inputs=[mode],
            outputs=[chat_response, generate_prompt, results, chart,
                     *add_buttons, formatted_prompt_view],
        )

    return app


def main():
    print("Starting Token Probability Analyzer…")
    model_path = os.environ.get("GGUF_MODEL_PATH", DEFAULT_MODEL_PATH)
    if model_path and Path(model_path).exists():
        try:
            load_model(model_path, DEFAULT_CONTEXT_SIZE, DEFAULT_GPU_LAYERS)
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
    else:
        print("⚠️ No model loaded — configure in the Model Configuration tab")

    port = os.environ.get("GRADIO_SERVER_PORT")
    app = create_gradio_interface()
    launch_kwargs = dict(server_name="127.0.0.1", share=False, theme=gr.themes.Soft())
    if port:
        launch_kwargs["server_port"] = int(port)
    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
