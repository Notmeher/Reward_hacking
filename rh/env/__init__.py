def make_env(task, cond, sandbox, cfg: dict | None = None):
    cfg = cfg or {}
    if task.kind == "swebench":
        from rh.swe.env import SweEnv

        return SweEnv(task, cond, cfg.get("swe", {}), cmd_timeout=float(cfg.get("cmd_timeout", 60)))
    from rh.env.code_env import CodeEnv

    return CodeEnv(task, cond, sandbox, cmd_timeout=float(cfg.get("cmd_timeout", 20)),
                   output_limit=int(cfg.get("output_limit", 6000)))
