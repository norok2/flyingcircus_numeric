#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: code that is now deprecated but can still be useful for legacy scripts.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import sys  # System-specific parameters and functions
import math  # Mathematical functions
import functools  # Higher-order functions and operations on callable objects
import doctest  # Test interactive Python examples
import string  # Common string operations
import itertools  # Functions creating iterators for efficient looping

from flyingcircus import INFO, PATH
from flyingcircus import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from flyingcircus import elapsed, report
from flyingcircus import msg, dbg, fmt, fmtm
from flyingcircus import HAS_JIT, jit


# ======================================================================
def i_stack(
        arrs,
        num,
        axis=-1):
    """
    Stack an iterable of arrays of the same size along a specific axis.

    This is equivalent to `numpy.stack()` but fetches the arrays from an
    iterable instead.

    This is useful for reducing the memory footprint of stacking compared
    to `numpy.stack()` and similar functions, since it does not require
    a full sequence of the data to reside in memory.

    Args:
        arrs (Iterable[ndarray]): The (N-1)-dim arrays to stack.
            These must have the same shape.
        num (int): The number of arrays to stack.
        axis (int): Direction along which to stack the arrays.
            Supports also negative values.

    Returns:
        result (ndarray): The concatenated N-dim array.
    """
    iter_arrs = iter(arrs)
    arr = next(iter_arrs)
    ndim = arr.ndim + 1
    assert(-ndim <= axis < ndim)
    axis %= ndim
    base_shape = arr.shape
    shape = base_shape[:axis] + (num,) + base_shape[axis:]
    result = np.empty(shape, dtype=arr.dtype)
    slicing = tuple(
        slice(None) if j != axis else 0 for j, d in enumerate(shape))
    result[slicing] = arr
    for i, arr in enumerate(iter_arrs, 1):
        slicing = tuple(
            slice(None) if j != axis else i for j, d in enumerate(shape))
        result[slicing] = arr
    return result


# ======================================================================
def i_split(arr, axis=-1):
    """
    Split an array along a specific axis into a list of arrays

    This is a generator version of `numpy.split()`.

    Args:
        arr (ndarray): The N-dim array to split.
        axis (int): Direction for the splitting of the array.

    Yields:
        arr (ndarray): The next (N-1)-dim array from the splitting.
            All yielded arrays have the same shape.
    """
    assert(-arr.ndim <= axis < arr.ndim)
    axis %= arr.ndim
    for i in range(arr.shape[axis]):
        slicing = tuple(
            slice(None) if j != axis else i for j, d in enumerate(arr.shape))
        yield arr[slicing]


# ======================================================================
def slice_array(
        arr,
        axis=0,
        index=None):
    """
    Slice a (N-1)-dim sub-array from an N-dim array.

    DEPRECATED! (Use advanced `numpy` slicing instead!)

    Args:
        arr (np.ndarray): The input N-dim array
        axis (int): The slicing axis.
        index (int): The slicing index.
            If None, mid-value is taken.

    Returns:
        sliced (np.ndarray): The sliced (N-1)-dim sub-array

    Raises:
        ValueError: if index is out of bounds

    Examples:
        >>> arr = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        >>> slice_array(arr, 2, 1)
        array([[ 1,  5,  9],
               [13, 17, 21]])
        >>> slice_array(arr, 1, 2)
        array([[ 8,  9, 10, 11],
               [20, 21, 22, 23]])
        >>> slice_array(arr, 0, 0)
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> slice_array(arr, 0, 1)
        array([[12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]])
    """
    # initialize slice index
    slab = [slice(None)] * arr.ndim
    # ensure index is meaningful
    if index is None:
        index = np.int(arr.shape[axis] / 2.0)
    # check index
    if (index >= arr.shape[axis]) or (index < 0):
        raise ValueError('Invalid array index in the specified direction')
    # determine slice index
    slab[axis] = index
    # slice the array
    return arr[tuple(slab)]


# ======================================================================
def sequence(
        start,
        stop,
        step=None,
        precision=None):
    """
    Generate a sequence that steps linearly from start to stop.

    Args:
        start (int|float): The starting value.
        stop (int|float): The final value.
            This value is present in the resulting sequence only if the step is
            a multiple of the interval size.
        step (int|float): The step value.
            If None, it is automatically set to unity (with appropriate sign).
        precision (int): The number of decimal places to use for rounding.
            If None, this is estimated from the `step` paramenter.

    Yields:
        item (int|float): the next element of the sequence.

    Examples:
        >>> list(sequence(0, 1, 0.1))
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> list(sequence(0, 1, 0.3))
        [0.0, 0.3, 0.6, 0.9]
        >>> list(sequence(0, 10, 1))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(sequence(0.4, 4.6, 0.72))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0]
        >>> list(sequence(0.4, 4.72, 0.72, 2))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0, 4.72]
        >>> list(sequence(0.4, 4.72, 0.72, 4))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0, 4.72]
        >>> list(sequence(0.4, 4.72, 0.72, 1))
        [0.4, 1.1, 1.8, 2.6, 3.3, 4.0, 4.7]
        >>> list(sequence(0.73, 5.29))
        [0.73, 1.73, 2.73, 3.73, 4.73]
        >>> list(sequence(-3.5, 3.5))
        [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        >>> list(sequence(3.5, -3.5))
        [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5]
        >>> list(sequence(10, 1, -1))
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        >>> list(sequence(10, 1, 1))
        []
        >>> list(sequence(10, 20, 10))
        [10, 20]
        >>> list(sequence(10, 20, 15))
        [10]
    """
    if step is None:
        step = 1 if stop > start else -1
    if precision is None:
        precision = fc.guess_decimals(step)
    for i in range(int(round(stop - start, precision + 1) / step) + 1):
        item = start + i * step
        if precision:
            item = round(item, precision)
        yield item


# ======================================================================
def accumulate(
        items,
        func=lambda x, y: x + y):
    """
    Cumulatively apply the specified function to the elements of the list.

    Args:
        items (Iterable): The items to process.
        func (callable): func(x,y) -> z
            The function applied cumulatively to the first n items of the list.
            Defaults to cumulative sum.

    Returns:
        lst (list): The cumulative list.

    See Also:
        itertools.accumulate.
    Examples:
        >>> accumulate(list(range(5)))
        [0, 1, 3, 6, 10]
        >>> accumulate(list(range(5)), lambda x, y: (x + 1) * y)
        [0, 1, 4, 15, 64]
        >>> accumulate([1, 2, 3, 4, 5, 6, 7, 8], lambda x, y: x * y)
        [1, 2, 6, 24, 120, 720, 5040, 40320]
    """
    return [
        functools.reduce(func, list(items)[:i + 1])
        for i in range(len(items))]


# ======================================================================
def cyclic_padding_loops(
        arr,
        shape,
        offsets):
    """
    Generate a cyclical padding of an array to a given shape with offsets.

    Implemented using single element loops.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The cyclic padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(cyclic_padding_loops(arr, (4, 5), (1, 1)))
        [[5 6 4 5 6]
         [2 3 1 2 3]
         [5 6 4 5 6]
         [2 3 1 2 3]]
    """
    shape = fc.auto_repeat(shape, arr.ndim, check=True)
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        (int(round((new_dim - dim) * offset))
         if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    result = np.zeros(shape, dtype=arr.dtype)
    for ij in itertools.product(*tuple(range(dim) for dim in result.shape)):
        slicing = tuple(
            (i + offset) % dim
            for i, offset, dim in zip(ij, offsets, arr.shape))
        result[ij] = arr[slicing]
    return result


# ======================================================================
def symmetric_padding_loops(
        arr,
        shape,
        offsets):
    """
    Generate a symmetrical padding of an array to a given shape with offsets.

    Implemented using single element loops.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The symmetric padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(symmetric_padding_loops(arr, (4, 5), (-1, -1)))
        [[6 6 5 4 4]
         [6 6 5 4 4]
         [3 3 2 1 1]
         [3 3 2 1 1]]
    """
    shape = fc.auto_repeat(shape, arr.ndim, check=True)
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        (int(round((new_dim - dim) * offset))
         if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    result = np.zeros(shape, dtype=arr.dtype)
    for ij in itertools.product(*tuple(range(dim) for dim in result.shape)):
        slicing = tuple(
            (i + offset) % dim
            if not ((i + offset) // dim % 2) else
            (dim - 1 - i - offset) % dim
            for i, offset, dim in zip(ij, offsets, arr.shape))
        result[ij] = arr[slicing]
    return result


# ======================================================================
def cyclic_padding_tile(
        arr,
        shape,
        offsets):
    """
    Generate a cyclical padding of an array to a given shape with offsets.

    Implemented using single element loops.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The cyclic padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(cyclic_padding_tile(arr, (4, 5), (1, 1)))
        [[5 6 4 5 6]
         [2 3 1 2 3]
         [5 6 4 5 6]
         [2 3 1 2 3]]
    """
    shape = fc.auto_repeat(shape, arr.ndim, check=True)
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        (int(round((new_dim - dim) * offset))
         if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    tiling = tuple(
        new_dim // dim + (1 if new_dim % dim else 0) + (1 if offset else 0)
        for offset, dim, new_dim in zip(offsets, arr.shape, shape))
    result = np.tile(arr, tiling)
    slicing = tuple(
        slice(offset, offset + new_dim)
        for offset, new_dim in zip(offsets, shape))
    return result[slicing]


# ======================================================================
def cyclic_padding_pad(
        arr,
        shape,
        offsets):
    """
    Generate a cyclical padding of an array to a given shape with offsets.

    Implemented using padding.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The cyclic padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(cyclic_padding_pad(arr, (4, 5), (1, 1)))
        [[5 6 4 5 6]
         [2 3 1 2 3]
         [5 6 4 5 6]
         [2 3 1 2 3]]
    """
    shape = fc.auto_repeat(shape, arr.ndim, check=True)
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        -(int(round((new_dim - dim) * offset))
          if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    width = tuple(
        (0, new_dim - dim)
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    result = np.pad(np.roll(arr, offsets, range(arr.ndim)), width, mode='wrap')
    return result


# ======================================================================
def cyclic_padding_slicing(
        arr,
        shape,
        offsets):
    """
    Generate a cyclical padding of an array to a given shape with offsets.

    Implemented using slicing.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The cyclic padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(cyclic_padding_slicing(arr, (4, 5), (1, 1)))
        [[5 6 4 5 6]
         [2 3 1 2 3]
         [5 6 4 5 6]
         [2 3 1 2 3]]
    """
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        (int(round((new_dim - dim) * offset))
         if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    views = tuple(
        tuple(
            slice(max(0, dim * i - offset), dim * (i + 1) - offset)
            for i in range((new_dim + offset) // dim))
        + (slice(dim * ((new_dim + offset) // dim) - offset, new_dim),)
        for offset, dim, new_dim in zip(offsets, arr.shape, shape))
    views = tuple(
        tuple(slice_ for slice_ in view if slice_.start < slice_.stop)
        for view in views)
    result = np.zeros(shape, dtype=arr.dtype)
    for view in itertools.product(*views):
        slicing = tuple(
            slice(None)
            if slice_.stop - slice_.start == dim else (
                slice(offset, offset + (slice_.stop - slice_.start))
                if slice_.start == 0 else
                slice(0, (slice_.stop - slice_.start)))
            for slice_, offset, dim in zip(view, offsets, arr.shape))
        result[view] = arr[slicing]
    return result


# ======================================================================
def frame(
        arr,
        borders=0.05,
        background=0.0,
        use_longest=True):
    """
    Add a background frame to an array specifying the borders.

    Note that this is similar to `fc.extra.padding()` but it is significantly
    faster, although less flexible.

    Args:
        arr (np.ndarray): The input array.
        borders (int|float|Iterable[int|float]): The border size(s).
            If int, this is in units of pixels.
            If float, this is proportional to the initial array shape.
            If int or float, uses the same value for all dimensions.
            If Iterable, the size must match `arr` dimensions.
            If 'use_longest' is True, use the longest dimension for the
            calculations.
        background (int|float): The background value to be used for the frame.
        use_longest (bool): Use longest dimension to get the border size.

    Returns:
        result (np.ndarray): The result array with added borders.

    See Also:
        - flyingcircus.extra.reframe()
        - flyingcircus.extra.padding()
    """
    borders = base.auto_repeat(borders, arr.ndim)
    if any(borders) < 0:
        raise ValueError('relative border cannot be negative')
    if isinstance(borders[0], float):
        if use_longest:
            dim = max(arr.shape)
            borders = [round(border * dim) for border in borders]
        else:
            borders = [
                round(border * dim) for dim, border in zip(arr.shape, borders)]
    result = np.full(
        [dim + 2 * border for dim, border in zip(arr.shape, borders)],
        background, dtype=arr.dtype)
    inner = tuple(
        slice(border, border + dim, None)
        for dim, border in zip(arr.shape, borders))
    result[inner] = arr
    return result


# ======================================================================
def reframe(
        arr,
        new_shape,
        position=0.5,
        background=0.0):
    """
    Add a frame to an array by centering the input array into a new shape.

    Args:
        arr (np.ndarray): The input array.
        new_shape (int|Iterable[int]): The shape of the output array.
            If int, uses the same value for all dimensions.
            If Iterable, the size must match `arr` dimensions.
            Additionally, each value of `new_shape` must be greater than or
            equal to the corresponding dimensions of `arr`.
        position (int|float|Iterable[int|float]): Position within new shape.
            Determines the position of the array within the new shape.
            If int or float, it is considered the same in all dimensions,
            otherwise its length must match the number of dimensions of the
            array.
            If int or Iterable of int, the values are absolute and must be
            less than or equal to the difference between the shape of the array
            and the new shape.
            If float or Iterable of float, the values are relative and must be
            in the [0, 1] range.
        background (int|float): The background value to be used for the frame.

    Returns:
        result (np.ndarray): The result array with added borders.

    Raises:
        IndexError: input and output shape sizes must match.
        ValueError: output shape cannot be smaller than the input shape.

    See Also:
        - flyingcircus.extra.frame()
        - flyingcircus.extra.padding()

    Examples:
        >>> arr = np.ones((2, 3))
        >>> reframe(arr, (4, 5))
        array([[0., 0., 0., 0., 0.],
               [0., 1., 1., 1., 0.],
               [0., 1., 1., 1., 0.],
               [0., 0., 0., 0., 0.]])
        >>> reframe(arr, (4, 5), 0)
        array([[1., 1., 1., 0., 0.],
               [1., 1., 1., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        >>> reframe(arr, (4, 5), (2, 0))
        array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [1., 1., 1., 0., 0.],
               [1., 1., 1., 0., 0.]])
        >>> reframe(arr, (4, 5), (0.0, 1.0))
        array([[0., 0., 1., 1., 1.],
               [0., 0., 1., 1., 1.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
    """
    new_shape = fc.auto_repeat(new_shape, arr.ndim, check=True)
    position = fc.auto_repeat(position, arr.ndim, check=True)
    if any(old > new for old, new in zip(arr.shape, new_shape)):
        raise ValueError('new shape cannot be smaller than the old one.')
    position = tuple(
        int(round((new - old) * x_i)) if isinstance(x_i, float) else x_i
        for old, new, x_i in zip(arr.shape, new_shape, position))
    if any(old + x_i > new
           for old, new, x_i in zip(arr.shape, new_shape, position)):
        raise ValueError(
            'Incompatible `new_shape`, `array shape` and `position`.')
    result = np.full(new_shape, background)
    inner = tuple(
        slice(offset, offset + dim, None)
        for dim, offset in zip(arr.shape, position))
    result[inner] = arr
    return result


# ======================================================================
def transparent_compression(func):
    """WIP"""

    def _wrapped(fp):
        from importlib import import_module


        zip_module_names = "gzip", "bz2"
        fallback_module_name = "builtins"
        open_module_names = zip_module_names + (fallback_module_name,)
        for open_module_name in open_module_names:
            try:
                open_module = import_module(open_module_name)
                tmp_fp = open_module.open(fp, "rb")
                tmp_fp.read(1)
            except (OSError, IOError, AttributeError, ImportError) as e:
                if open_module_name is fallback_module_name:
                    raise e
            else:
                tmp_fp.seek(0)
                fp = tmp_fp
                break
        return func(fp=fp)

    return _wrapped


# ======================================================================
def ssim(
        arr1,
        arr2,
        arr_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the structure similarity index, SSIM.

    This is defined as: SSIM = (lum ** alpha) * (con ** beta) * (sti ** gamma)
     - lum is a measure of the luminosity, with exp. weight alpha
     - con is a measure of the contrast, with exp. weight beta
     - sti is a measure of the structural information, with exp. weight gamma

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors. Must be 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors. Must
        be 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim (float): The structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    if arr_interval is None:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    range_size = np.ptp(arr_interval)
    cc = [(k * range_size) ** 2 for k in kk]
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1 = np.std(arr1)
    sigma2 = np.std(arr2)
    sigma12 = np.sum((arr1 - mu1) * (arr2 - mu2)) / (arr1.size - 1)
    ff = [
        (2 * mu1 * mu2 + cc[0]) / (mu1 ** 2 + mu2 ** 2 + cc[0]),
        (2 * sigma1 * sigma2 + cc[1]) / (sigma1 ** 2 + sigma2 ** 2 + cc[1]),
        (sigma12 + cc[2]) / (sigma1 * sigma2 + cc[2])
    ]
    return np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)


# ======================================================================
def ssim_map(
        arr1,
        arr2,
        filter_sizes=5,
        sigmas=1.5,
        arr_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the local structure similarity index map.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        filter_sizes (tuple[int]|int): The size of the filter in px.
            If a single value is given, is is assumed to be equal in all dims.
        sigmas (tuple[float]|float): The sigma of the gaussian kernel in px.
            If a single value is given, it is assumed to be equal in all dims.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors.
            Must be of length 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors.
            Must be of length 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim_arr (np.ndarray): The local structure similarity index map
        ssim (float): The global (mean) structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    if arr_interval is None:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    range_size = np.ptp(arr_interval)
    ndim = arr1.ndim
    arr_filter = fc.extra.gaussian_nd(filter_sizes, sigmas, 0.5, ndim, True)
    convolve = sp.signal.fftconvolve
    mu1 = convolve(arr1, arr_filter, 'same')
    mu2 = convolve(arr2, arr_filter, 'same')
    mu1_mu1 = mu1 ** 2
    mu2_mu2 = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sg1_sg1 = convolve(arr1 ** 2, arr_filter, 'same') - mu1_mu1
    sg2_sg2 = convolve(arr2 ** 2, arr_filter, 'same') - mu2_mu2
    sg12 = convolve(arr1 * arr2, arr_filter, 'same') - mu1_mu2
    cc = [(k * range_size) ** 2 for k in kk]
    # determine whether to use the simplified expression
    if all(aa) == 1 and 2 * cc[2] == cc[1]:
        ssim_arr = ((2 * mu1_mu2 + cc[0]) * (2 * sg12 + cc[1])) / (
                (mu1_mu1 + mu2_mu2 + cc[0]) * (sg1_sg1 + sg2_sg2 + cc[1]))
    else:
        sg1 = np.sqrt(np.abs(sg1_sg1))
        sg2 = np.sqrt(np.abs(sg2_sg2))
        ff = [
            (2 * mu1_mu2 + cc[0]) / (mu1_mu1 + mu2_mu2 + cc[0]),
            (2 * sg1 * sg2 + cc[1]) / (sg1_sg1 + sg2_sg2 + cc[1]),
            (sg12 + cc[2]) / (sg1 * sg2 + cc[2])
        ]
        ssim_arr = np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)
    ssim_val = np.mean(ssim_arr)
    return ssim_arr, ssim_val


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()
