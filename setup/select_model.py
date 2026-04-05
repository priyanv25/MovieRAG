# Run this on your machine to find the best models:
# pip install mteb

import mteb

metas = mteb.get_model_metas()

# Filter: 200M-600M params, open-weight, retrieval-capable
candidates = [
    m for m in metas
    if m.n_parameters is not None
    and 200_000_000 <= m.n_parameters <= 600_000_000
    and m.open_weights
]

print(f"Found {len(candidates)} models in 200-600M range\n")
for m in sorted(candidates, key=lambda x: x.n_parameters):
    print(f"  {m.name:<50} {m.n_parameters/1e6:.0f}M")

