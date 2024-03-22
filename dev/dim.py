from tests.dummy_datasets import generate_corner_dataset
from src.antakia_core.compute.dim_reduction.dim_reduction import compute_projection

df = generate_corner_dataset(200)
proj = compute_projection(df[0], df[1], 3, 2, lambda *args: print(*args), verbose=True)
