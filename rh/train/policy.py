"""LoRA policy trainer for multi-turn GRPO on a single GPU (fp16, T4-friendly).

Rollouts come from a separate vLLM server; here we re-render each episode with the model's
chat template, mask everything except assistant turns, and take one on-policy gradient step
per iteration with loss = -sum(A * log p(assistant tokens)) / total_assistant_tokens.
"""
import logging
import math
import re
from pathlib import Path

log = logging.getLogger(__name__)
ASSISTANT_SPAN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


class PolicyTrainer:
    def __init__(self, model_path: str, lora: dict, lr: float = 1e-5, device: str = "cuda:0",
                 max_len: int = 10240, max_grad_norm: float = 1.0, logit_chunk: int = 1024,
                 assistant_pattern: str | None = None):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device, self.max_len = torch, device, max_len
        self.max_grad_norm, self.logit_chunk = max_grad_norm, logit_chunk
        self.span = re.compile(assistant_pattern, re.S) if assistant_pattern else ASSISTANT_SPAN
        self.tok = AutoTokenizer.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map={"": device},
                                                    attn_implementation="sdpa")
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        cfg = LoraConfig(r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
                         lora_dropout=float(lora.get("dropout", 0.0)), bias="none", task_type="CAUSAL_LM",
                         target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj",
                                                                    "gate_proj", "up_proj", "down_proj"]))
        self.model = get_peft_model(base, cfg)
        for p in self.model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.scaler = torch.amp.GradScaler("cuda")
        n = sum(p.numel() for p in self.params)
        log.info("trainable LoRA params: %.1fM", n / 1e6)

    def encode(self, messages: list, tools: list):
        torch = self.torch
        msgs = []
        for m in messages:
            m = dict(m)
            if m.get("content") is None:
                m["content"] = ""
            msgs.append(m)
        text = self.tok.apply_chat_template(msgs, tools=tools, tokenize=False)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        if len(ids) > self.max_len:
            return None
        spans = [(m.start(1), m.end(1)) for m in self.span.finditer(text)]
        mask = [any(s <= a < e for s, e in spans) for a, _ in enc["offset_mapping"]]
        if not any(mask):
            return None
        return torch.tensor(ids, device=self.device), torch.tensor(mask, device=self.device)

    def token_logprobs(self, ids, mask):
        torch = self.torch
        base = self.model.get_base_model()
        hidden = base.model(input_ids=ids[None]).last_hidden_state[0]
        pos = torch.nonzero(mask[1:]).squeeze(-1)
        targets = ids[pos + 1]
        out = []
        for i in range(0, len(pos), self.logit_chunk):
            h = hidden[pos[i : i + self.logit_chunk]]
            logits = base.lm_head(h).float()
            out.append(torch.log_softmax(logits, -1).gather(-1, targets[i : i + self.logit_chunk, None]).squeeze(-1))
        return torch.cat(out) if out else hidden.new_zeros(0, dtype=torch.float32)

    def step(self, batch: list[tuple[list, list, float]]) -> dict:
        """batch: (messages, tools, advantage). Zero-advantage items should be filtered by the caller."""
        torch = self.torch
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        enc = []
        for msgs, tools, adv in batch:
            e = self.encode(msgs, tools)
            if e is not None:
                enc.append((*e, adv))
        skipped = len(batch) - len(enc)
        if not enc:
            return {"loss": 0.0, "grad_norm": 0.0, "tokens": 0, "skipped": skipped, "stepped": False}
        total = sum(int(m[1:].sum()) for _, m, _ in enc)
        loss_sum, mean_lp = 0.0, 0.0
        for ids, mask, adv in enc:
            with torch.autocast("cuda", dtype=torch.float16):
                lp = self.token_logprobs(ids, mask)
            loss = -(adv * lp.sum()) / max(1, total)
            self.scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
            mean_lp += float(lp.detach().sum())
        self.scaler.unscale_(self.opt)
        gn = float(torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm))
        stepped = math.isfinite(gn)
        self.scaler.step(self.opt)
        self.scaler.update()
        torch.cuda.empty_cache()
        return {"loss": loss_sum, "grad_norm": gn, "tokens": total, "skipped": skipped, "stepped": stepped,
                "mean_token_logprob": mean_lp / max(1, total)}

    def save_adapter(self, path: Path) -> None:
        self.model.save_pretrained(str(path))
        self.tok.save_pretrained(str(path))

    def save_state(self, path: Path) -> None:
        from peft import get_peft_model_state_dict

        self.torch.save({"adapter": get_peft_model_state_dict(self.model), "opt": self.opt.state_dict(),
                         "scaler": self.scaler.state_dict()}, str(path))

    def load_state(self, path: Path) -> None:
        from peft import set_peft_model_state_dict

        st = self.torch.load(str(path), map_location=self.device, weights_only=False)
        set_peft_model_state_dict(self.model, st["adapter"])
        self.opt.load_state_dict(st["opt"])
        self.scaler.load_state_dict(st["scaler"])
