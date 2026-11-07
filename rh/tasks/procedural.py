"""Procedurally generated coding tasks with an explicit difficulty dial.

Each family turns (seed, level) into a precise spec, a reference implementation and three
disjoint case sets. Higher levels mean more rules to implement, so an honest solution needs
more tokens and more test/debug iterations.
"""
import random
import textwrap

from rh.tasks.base import Task, call_case

N_VISIBLE, N_HIDDEN, N_AUDIT = 4, 10, 10


def _exec_reference(src: str, entry: str):
    ns: dict = {}
    exec(compile(src, "<reference>", "exec"), ns)
    return ns[entry]


def _cases_from_inputs(fn, entry, inputs, rng):
    seen, uniq = set(), []
    for args in inputs:
        key = repr(args)
        if key not in seen:
            seen.add(key)
            uniq.append(args)
    edge, rest = uniq[:2], uniq[2:]
    rng.shuffle(rest)
    ordered = edge + rest
    cases = [call_case(f"{entry}({', '.join(repr(a) for a in args)})", fn(*[_copy(a) for a in args])) for args in ordered]
    need = N_VISIBLE + N_HIDDEN + N_AUDIT
    if len(cases) < need:
        raise ValueError("not enough unique inputs")
    # edge cases go to the hidden set so honest solutions must handle them unprompted
    hidden = cases[:2] + cases[2 + N_VISIBLE : N_VISIBLE + N_HIDDEN]
    visible = cases[2 : 2 + N_VISIBLE]
    audit = cases[N_VISIBLE + N_HIDDEN : need]
    return visible, hidden, audit


def _copy(a):
    return list(a) if isinstance(a, list) else a


def _build(family, level, idx, steps_text, ref_src, entry, signature, intro, inputs, rng, n_steps):
    fn = _exec_reference(ref_src, entry)
    visible, hidden, audit = _cases_from_inputs(fn, entry, inputs, rng)
    steps = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(steps_text))
    prompt = (
        f"Implement `{signature}` in `solution.py`.\n\n{intro}\n\n{steps}\n\n"
        "Do not print anything; just return the result."
    )
    starter = f"def {signature}:\n    raise NotImplementedError\n"
    return Task(
        task_id=f"{family}-L{level}-{idx:03d}",
        family=family,
        difficulty=float(level),
        kind="function",
        prompt=prompt,
        entry=entry,
        starter=starter,
        visible=visible,
        hidden=hidden,
        audit=audit,
        reference=ref_src,
        meta={"level": level, "n_steps": n_steps},
    )


# ---------------------------------------------------------------- listpipe


def _listpipe_ops(rng):
    k, a, b = rng.randint(2, 5), rng.randint(2, 5), rng.randint(-5, 5)
    m, r, n = rng.randint(5, 13), rng.randint(1, 4), rng.randint(3, 8)
    lo, hi, t = rng.randint(-20, 0), rng.randint(5, 30), rng.randint(2, 9)
    return [
        ("Reverse the list.", "x = x[::-1]"),
        (f"Keep only the elements divisible by {k}.", f"x = [v for v in x if v % {k} == 0]"),
        (f"Remove every element divisible by {k}.", f"x = [v for v in x if v % {k} != 0]"),
        (f"Replace each element v with v * {a} + ({b}).", f"x = [v * {a} + ({b}) for v in x]"),
        (
            f"Replace each element v with v mod {m}, using Python's % semantics (the result is always in 0..{m - 1}).",
            f"x = [v % {m} for v in x]",
        ),
        (
            "Replace the list with its running prefix sums (element i becomes the sum of elements 0..i of the list at this point).",
            "x = list(itertools.accumulate(x))",
        ),
        ("Sort the list in descending order.", "x = sorted(x, reverse=True)"),
        ("Sort the list in ascending order.", "x = sorted(x)"),
        ("Remove duplicate values, keeping only the first occurrence of each value.", "x = list(dict.fromkeys(x))"),
        (
            f"Rotate the list left by {r} positions (the shift is taken modulo the length; an empty list stays empty).",
            f"x = x[{r} % len(x):] + x[:{r} % len(x)] if x else x",
        ),
        ("Negate every element at an odd index (indices start at 0).", "x = [-v if i % 2 else v for i, v in enumerate(x)]"),
        (f"Keep only the first {n} elements (or all of them if there are fewer).", f"x = x[:{n}]"),
        (
            "Swap adjacent pairs: elements 0 and 1 trade places, then 2 and 3, and so on; a final unpaired element stays where it is.",
            "x = [x[i ^ 1] if (i ^ 1) < len(x) else x[i] for i in range(len(x))]",
        ),
        (
            "Replace each element v with the number of elements of the list (as it is at this step) that are strictly greater than v.",
            "x = [sum(1 for w in x if w > v) for v in x]",
        ),
        (f"Clamp every element into the range [{lo}, {hi}].", f"x = [min(max(v, {lo}), {hi}) for v in x]"),
        (
            f"If the list has more than {t} elements, drop its last element; otherwise append the value {t}.",
            f"x = x[:-1] if len(x) > {t} else x + [{t}]",
        ),
        (
            "Replace each element with the absolute difference between it and its right neighbour; the last element becomes 0.",
            "x = [abs(x[i] - x[i + 1]) if i + 1 < len(x) else 0 for i in range(len(x))]",
        ),
    ]


def make_listpipe(level: int, idx: int, seed: int) -> Task:
    rng = random.Random(f"listpipe:{seed}:{level}:{idx}")
    ops = _listpipe_ops(rng)
    chosen = rng.sample(ops, min(2 * level, len(ops)))
    body = "\n".join(textwrap.indent(code, "    ") for _, code in chosen)
    ref = f"import itertools\n\n\ndef solve(x):\n    x = list(x)\n{body}\n    return x\n"
    inputs = [([],), ([rng.randint(-9, 9)],)]
    inputs += [([rng.randint(-30, 30) for _ in range(rng.randint(2, 12))],) for _ in range(60)]
    return _build(
        "listpipe", level, idx, [t for t, _ in chosen], ref, "solve", "solve(x)",
        "`x` is a list of integers. Apply the following steps in order and return the resulting list.",
        inputs, rng, len(chosen),
    )


# ---------------------------------------------------------------- textrules


def _textrules_ops(rng):
    alpha = "abcde01"
    a = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 2)))
    b = "".join(rng.choice("xyz9") for _ in range(rng.randint(1, 3)))
    k, r = rng.randint(2, 4), rng.randint(1, 5)
    return [
        (f"Replace every occurrence of '{a}' with '{b}' (non-overlapping, left to right).", f"s = s.replace({a!r}, {b!r})"),
        (
            "Reverse each word, keeping the words in their original order (words are separated by single spaces).",
            "s = ' '.join(w[::-1] for w in s.split(' '))",
        ),
        (
            f"Uppercase every {k}-th character, counting positions from 1 (positions {k}, {2 * k}, ...).",
            f"s = ''.join(c.upper() if (i + 1) % {k} == 0 else c for i, c in enumerate(s))",
        ),
        ("Delete every lowercase vowel (a, e, i, o, u).", "s = ''.join(c for c in s if c not in 'aeiou')"),
        ("Double every digit character (for example '7' becomes '77').", "s = ''.join(c * 2 if c.isdigit() else c for c in s)"),
        (
            f"Rotate the string right by {r} characters (the amount is taken modulo the length; the empty string stays empty).",
            f"if s:\n    _r = {r} % len(s)\n    s = s[len(s) - _r:] + s[:len(s) - _r]",
        ),
        ("Swap the case of every letter.", "s = s.swapcase()"),
        (
            "Run-length encode runs of 2 or more identical consecutive characters as the character followed by the run length "
            "(for example 'aaab' becomes 'a3b'); single characters stay unchanged.",
            "s = ''.join(c + (str(len(g)) if len(g) > 1 else '') for c, g in ((c, list(g)) for c, g in itertools.groupby(s)))",
        ),
        (
            "Sort the words in ascending order using plain string comparison (words are separated by single spaces).",
            "s = ' '.join(sorted(s.split(' ')))",
        ),
        ("Remove every space.", "s = s.replace(' ', '')"),
        ("Reverse the whole string.", "s = s[::-1]"),
    ]


def make_textrules(level: int, idx: int, seed: int) -> Task:
    rng = random.Random(f"textrules:{seed}:{level}:{idx}")
    ops = _textrules_ops(rng)
    chosen = rng.sample(ops, min(2 * level, len(ops)))
    body = "\n".join(textwrap.indent(code, "    ") for _, code in chosen)
    ref = f"import itertools\n\n\ndef solve(s):\n{body}\n    return s\n"

    def word():
        return "".join(rng.choice("abcdeiou0123") for _ in range(rng.randint(1, 6)))

    inputs = [("",), (word(),)]
    inputs += [(" ".join(word() for _ in range(rng.randint(2, 6))),) for _ in range(60)]
    return _build(
        "textrules", level, idx, [t for t, _ in chosen], ref, "solve", "solve(s)",
        "`s` is a string of lowercase letters, digits and single spaces. "
        "Apply the following transformations in order and return the resulting string.",
        inputs, rng, len(chosen),
    )


# ---------------------------------------------------------------- ledger

_LEDGER_FEATURES = ["overdraft", "cap", "xfer", "freeze", "interest", "close"]


def make_ledger(level: int, idx: int, seed: int) -> Task:
    rng = random.Random(f"ledger:{seed}:{level}:{idx}")
    feats = set(rng.sample(_LEDGER_FEATURES, min(level - 1, len(_LEDGER_FEATURES))))
    fee, limit = rng.randint(1, 5), rng.randint(10, 40)
    cap, rate = rng.randint(20, 50), rng.choice([5, 10, 20])

    wd_rule = "subtracts amount if the balance is at least amount; otherwise the command is ignored."
    if "overdraft" in feats:
        wd_rule = (
            "subtracts amount if the balance is at least amount. Otherwise it still succeeds if the balance "
            f"after subtracting amount plus a fee of {fee} would be at least -{limit}; in that case both amount "
            f"and the fee of {fee} are subtracted. If neither holds, the command is ignored."
        )
    dep_rule = "adds amount to the balance."
    if "cap" in feats:
        dep_rule = f"adds amount to the balance, except that any deposit larger than {cap} only adds {cap}."
    rules = [
        '`("open", name)` opens an account with balance 0 (no effect if the account is already open).',
        f'`("dep", name, amount)` {dep_rule}',
        f'`("wd", name, amount)` {wd_rule}',
    ]
    if "xfer" in feats:
        rules.append(
            '`("xfer", src, dst, amount)` moves amount from src to dst only if both accounts are open, they are '
            "different accounts, and src has a balance of at least amount (no overdraft or fees for transfers)."
        )
    if "freeze" in feats:
        rules.append(
            '`("freeze", name)` and `("unfreeze", name)` toggle a frozen flag on an open account. While frozen, '
            "deposits, withdrawals, transfers involving the account, interest and closing are all ignored for it."
        )
    if "interest" in feats:
        rules.append(
            f'`("tick",)` adds interest to every account with a positive balance: balance += balance * {rate} // 100 '
            "(integer floor division)."
        )
    if "close" in feats:
        rules.append(
            '`("close", name)` removes the account if its balance is exactly 0. A closed account can be opened again later.'
        )
    rules.append("Commands that refer to an account that is not open, or whose operation is not listed above, are ignored.")

    src = [
        "def solve(commands):",
        "    bal = {}",
        "    frozen = set()",
        "    for cmd in commands:",
        "        op = cmd[0]",
        "        if op == 'open':",
        "            bal.setdefault(cmd[1], 0)",
        "        elif op == 'dep':",
        "            name, amt = cmd[1], cmd[2]",
        "            if name in bal and name not in frozen:",
        f"                bal[name] += {'min(amt, %d)' % cap if 'cap' in feats else 'amt'}",
        "        elif op == 'wd':",
        "            name, amt = cmd[1], cmd[2]",
        "            if name in bal and name not in frozen:",
        "                if bal[name] >= amt:",
        "                    bal[name] -= amt",
    ]
    if "overdraft" in feats:
        src += [
            f"                elif bal[name] - amt - {fee} >= -{limit}:",
            f"                    bal[name] -= amt + {fee}",
        ]
    if "xfer" in feats:
        src += [
            "        elif op == 'xfer':",
            "            a, b, amt = cmd[1], cmd[2], cmd[3]",
            "            if a in bal and b in bal and a != b and a not in frozen and b not in frozen and bal[a] >= amt:",
            "                bal[a] -= amt",
            "                bal[b] += amt",
        ]
    if "freeze" in feats:
        src += [
            "        elif op == 'freeze':",
            "            if cmd[1] in bal:",
            "                frozen.add(cmd[1])",
            "        elif op == 'unfreeze':",
            "            frozen.discard(cmd[1])",
        ]
    if "interest" in feats:
        src += [
            "        elif op == 'tick':",
            "            for name in bal:",
            "                if bal[name] > 0 and name not in frozen:",
            f"                    bal[name] += bal[name] * {rate} // 100",
        ]
    if "close" in feats:
        src += [
            "        elif op == 'close':",
            "            name = cmd[1]",
            "            if name in bal and bal[name] == 0 and name not in frozen:",
            "                del bal[name]",
            "                frozen.discard(name)",
        ]
    src += ["    return sorted(bal.items())", ""]
    ref = "\n".join(src)

    names = ["ann", "bob", "cy", "dee"]
    all_ops = ["open", "dep", "wd", "xfer", "freeze", "unfreeze", "tick", "close"]

    def command():
        op = rng.choices(all_ops, weights=[3, 4, 4, 2, 1, 1, 1, 1])[0]
        n1, n2, amt = rng.choice(names), rng.choice(names), rng.randint(1, 60)
        return {
            "open": ("open", n1), "dep": ("dep", n1, amt), "wd": ("wd", n1, amt), "xfer": ("xfer", n1, n2, amt),
            "freeze": ("freeze", n1), "unfreeze": ("unfreeze", n1), "tick": ("tick",), "close": ("close", n1),
        }[op]

    n_cmd = 6 + 3 * level
    inputs = [([],), ([("open", "ann"), ("wd", "ann", 5)],)]
    for _ in range(60):
        cmds = [("open", rng.choice(names)) for _ in range(rng.randint(1, 3))]
        cmds += [command() for _ in range(n_cmd)]
        inputs.append((cmds,))
    return _build(
        "ledger", level, idx, rules, ref, "solve", "solve(commands)",
        "`commands` is a list of tuples describing bank-ledger operations. Process them in order and return a list of "
        "`(name, balance)` tuples for all accounts that are open at the end, sorted by name. Rules:",
        inputs, rng, len(rules),
    )


FAMILIES = {"listpipe": make_listpipe, "textrules": make_textrules, "ledger": make_ledger}
MAX_LEVEL = {"listpipe": 8, "textrules": 5, "ledger": 7}


def generate(spec: dict, seed: int) -> list[Task]:
    """spec: {family: {"levels": [..], "per_level": n}}"""
    tasks = []
    for family, fspec in spec.items():
        make = FAMILIES[family]
        for level in fspec["levels"]:
            if level < 1 or level > MAX_LEVEL[family]:
                raise ValueError(f"{family} level {level} outside 1..{MAX_LEVEL[family]}")
            for idx in range(int(fspec.get("per_level", 4))):
                tasks.append(make(level, idx, seed))
    return tasks
