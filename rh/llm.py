"""Single-shot (no tools) completions across providers, for monitors and judges."""


async def complete(spec: dict, system: str, user: str) -> str:
    p = spec["provider"]
    if p in ("azure", "vllm"):
        from rh.agents import azure_client, openai_compat_client

        if p == "azure":
            client = azure_client(spec.get("api_version"), spec.get("endpoint"))
            model = spec.get("deployment")
        else:
            client = openai_compat_client(spec["base_urls"][0], spec.get("api_key", "EMPTY"))
            model = spec["served_name"]
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **spec.get("params", {}))
        return resp.choices[0].message.content or ""
    if p == "anthropic":
        from rh.agents import anthropic_client

        resp = await anthropic_client().messages.create(**anthropic_params(spec, system, user))
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    raise ValueError(f"unsupported provider {p!r}")


def anthropic_params(spec: dict, system: str, user: str) -> dict:
    params = {"model": spec["model"], "max_tokens": int(spec.get("max_tokens", 4000)), "system": system,
              "messages": [{"role": "user", "content": user}]}
    extra = {}
    if spec.get("thinking"):
        extra["thinking"] = spec["thinking"]
    if spec.get("effort"):
        extra["output_config"] = {"effort": spec["effort"]}
    if extra:
        params["extra_body"] = extra
    return params
