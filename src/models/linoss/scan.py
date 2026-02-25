"""
Pure PyTorch associative scan (Blelloch algorithm).

Vendored from TritonLinOSS (MIT License).
This is the CPU/torch.compile fallback for the parallel scan.
On GPU with Triton, the Triton kernels are used instead.
"""
import torch
from torch.utils import _pytree as pytree
from typing import Callable, Any


def associative_scan(fn: Callable, elems: Any, reverse: bool = False, axis: int = 0):
    """
    Performs a scan with an associative binary operation, in parallel.
    PyTorch port of jax.lax.associative_scan.

    Args:
        fn: A callable implementing an associative binary operation r = fn(a, b).
        elems: A (possibly nested) pytree of tensors.
        reverse: If True, scan from end to start.
        axis: The axis along which to scan.
    """

    # Flatten the pytree (handles nested dicts/lists of tensors)
    elems_flat, tree_spec = pytree.tree_flatten(elems)

    if not elems_flat:
        raise ValueError("elems must not be empty")

    # Ensure all elements are tensors
    elems_flat = [torch.as_tensor(e) for e in elems_flat]

    # Canonicalize axis
    ndim = elems_flat[0].ndim
    axis = axis % ndim

    # Validate shapes
    num_elems = elems_flat[0].shape[axis]
    if not all(e.shape[axis] == num_elems for e in elems_flat[1:]):
        raise ValueError(
            f"All input arrays must have the same size {num_elems} along axis {axis}."
        )

    # Handle reverse scan
    if reverse:
        elems_flat = [torch.flip(e, [axis]) for e in elems_flat]

    # Helper to combine flattened inputs using the user function
    def combine(a_flat, b_flat):
        a = pytree.tree_unflatten(a_flat, tree_spec)
        b = pytree.tree_unflatten(b_flat, tree_spec)
        c = fn(a, b)
        c_flat, _ = pytree.tree_flatten(c)
        return c_flat

    # Helper to slice tensors generically along an axis
    def slice_at_axis(t, sl):
        idx = [slice(None)] * t.ndim
        idx[axis] = sl
        return t[tuple(idx)]

    # Helper to concatenate tensors along the axis
    def cat_at_axis(tensors):
        return torch.cat(tensors, dim=axis)

    # Recursive scan implementation (Blelloch algorithm)
    def _scan(elems_curr):
        n = elems_curr[0].shape[axis]

        if n < 2:
            return elems_curr

        # 1. Reduce phase: Combine adjacent pairs
        left_ops = [slice_at_axis(e, slice(0, -1, 2)) for e in elems_curr]
        right_ops = [slice_at_axis(e, slice(1, None, 2)) for e in elems_curr]

        reduced_elems = combine(left_ops, right_ops)

        # 2. Recursion
        odd_elems = _scan(reduced_elems)

        # 3. Down-sweep phase: Calculate even elements
        if n % 2 == 0:
            e_left = [slice_at_axis(e, slice(0, -1)) for e in odd_elems]
            e_right = [slice_at_axis(e, slice(2, None, 2)) for e in elems_curr]
        else:
            e_left = odd_elems
            e_right = [slice_at_axis(e, slice(2, None, 2)) for e in elems_curr]

        even_calc = combine(e_left, e_right)

        # Reconstruct evens: First element is always original first element
        first_elems = [slice_at_axis(e, slice(0, 1)) for e in elems_curr]

        even_elems = [cat_at_axis([f, calc]) for f, calc in zip(first_elems, even_calc)]

        # 4. Interleave evens and odds to form the final result
        return _interleave_lists(even_elems, odd_elems, axis)

    def _interleave_lists(evens, odds, ax):
        res = []
        for e, o in zip(evens, odds):
            n_e = e.shape[ax]
            n_o = o.shape[ax]

            if n_e == n_o:
                stacked = torch.stack([e, o], dim=ax + 1)
                new_shape = list(e.shape)
                new_shape[ax] = n_e + n_o
                res.append(stacked.reshape(new_shape))
            elif n_e == n_o + 1:
                e_part = slice_at_axis(e, slice(0, -1))
                last_e = slice_at_axis(e, slice(-1, None))
                stacked = torch.stack([e_part, o], dim=ax + 1)
                flat_shape = list(e_part.shape)
                flat_shape[ax] = n_e - 1 + n_o
                flattened = stacked.reshape(flat_shape)
                res.append(torch.cat([flattened, last_e], dim=ax))
            else:
                raise RuntimeError(
                    f"Shape mismatch in interleave: {e.shape} vs {o.shape}"
                )
        return res

    # Run the scan
    scanned_flat = _scan(elems_flat)

    # Un-reverse if needed
    if reverse:
        scanned_flat = [torch.flip(s, [axis]) for s in scanned_flat]

    return pytree.tree_unflatten(scanned_flat, tree_spec)
