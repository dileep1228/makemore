# makemore, from scratch

Character-level language models built by hand while working through Andrej
Karpathy's **makemore** videos, parts 1–5 of
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).

Heavily inspired by — and following — those videos. The design and the order
things are built in are Karpathy's; the original is at
[karpathy/makemore](https://github.com/karpathy/makemore), MIT licensed. This is
a learning reimplementation, typed out and debugged rather than copied.

Companion repo: [micrograd from scratch](https://github.com/dileep1228/micrograd),
the autograd engine underneath all of this.

## Progress

- **Part 1 — bigrams.** Counting bigrams into a 27×27 matrix, normalising rows
  into probability distributions, sampling with `torch.multinomial`, negative log
  likelihood as a loss, then the same model rebuilt as a single-layer neural net
  trained by gradient descent. Both approaches land on the same answer.
- Parts 2–5: to come.

## Data

`names.txt` — 32,033 names, from Karpathy's makemore repo.

## Running it

```bash
uv sync
uv run jupyter lab
```
