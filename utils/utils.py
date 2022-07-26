import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import ceil
from scipy.ndimage.filters import gaussian_filter
import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.transforms import Resize
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
    optional_import,
)

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()

def generate_label(input_lbl, num_classes, name, TEMPLATE):
    """
    Convert class index tensor to one hot encoding tensor with -1 (ignored).
    Args:
         input: A tensor of shape [bs, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    """
    shape = np.array(input_lbl.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    input_lbl = input_lbl.long()
    result = result.scatter_(1, input_lbl, 1)

    ## assign -1 to ignored
    B = result.shape[0]
    for b in range(B):
        organ_list = TEMPLATE[name[b][0:2]]
        for i in range(num_classes):
            if (i+1) not in organ_list:
                result[b, i] = -1
    # print(name)
    return result

def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def multi_net(net_list, img, task_id):
    # img = torch.from_numpy(img).cuda()

    padded_prediction = net_list[0](img, task_id)
    padded_prediction = F.sigmoid(padded_prediction)
    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img, task_id)
        padded_prediction_i = F.sigmoid(padded_prediction_i)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction#.cpu().data.numpy()

def predict_sliding(net_list, image, tile_size, classes, task_id):  # tile_size:32x256x256
    gaussian_importance_map = _get_gaussian(tile_size, sigma_scale=1. / 8)
    
    image_size = image.shape
    overlap = 1 / 2

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    # print("Need %i x %i x %i prediction tiles @ stride %i x %i px" % (tile_deps, tile_cols, tile_rows, strideD, strideHW))
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4]))#.astype(np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4]))#.astype(np.float32)
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[1]), 0)  # for very few rows y1 underflows

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img = torch.from_numpy(img).cuda()

                prediction1 = multi_net(net_list, img, task_id)
                prediction2 = torch.flip(multi_net(net_list, torch.flip(img, [2]), task_id), [2])
                prediction3 = torch.flip(multi_net(net_list, torch.flip(img, [3]), task_id), [3])
                prediction4 = torch.flip(multi_net(net_list, torch.flip(img, [4]), task_id), [4])
                prediction5 = torch.flip(multi_net(net_list, torch.flip(img, [2,3]), task_id), [2,3])
                prediction6 = torch.flip(multi_net(net_list, torch.flip(img, [2,4]), task_id), [2,4])
                prediction7 = torch.flip(multi_net(net_list, torch.flip(img, [3,4]), task_id), [3,4])
                prediction8 = torch.flip(multi_net(net_list, torch.flip(img, [2,3,4]), task_id), [2,3,4])
                prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5 + prediction6 + prediction7 + prediction8) / 8.
                prediction = prediction.cpu()
                
                prediction[:,:] *= gaussian_importance_map

                if isinstance(prediction, list):
                    shape = np.array(prediction[0].shape)
                    shape[0] = prediction[0].shape[0] * len(prediction)
                    shape = tuple(shape)
                    preds = torch.zeros(shape).cuda()
                    bs_singlegpu = prediction[0].shape[0]
                    for i in range(len(prediction)):
                        preds[i * bs_singlegpu: (i + 1) * bs_singlegpu] = prediction[i]
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += preds

                else:
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += gaussian_importance_map
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs

def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    task_id: Any,
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]  # type: ignore
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    # Perform predictions
    dict_key, output_image_list, count_map_list = None, [], []
    _initialized_ss = -1
    is_tensor_output = True  # whether the predictor's output is a tensor (instead of dict/tuple)

    # for each patch
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        N, _, _, _, _ = window_data.shape
        task_id_ = task_id.repeat(N)
        seg_prob_out = predictor(window_data, task_id_, *args, **kwargs)  # batched patch segmentation

        # convert seg_prob_out to tuple seg_prob_tuple, this does not allocate new memory.
        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                dict_key = sorted(seg_prob_out.keys())  # track predictor's output keys
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False

        # for each output in multi-output list
        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(device)  # BxCxMxNxP or BxCxMxN

            # compute zoom scale: out_roi_size/in_roi_size
            zoom_scale = []
            for axis, (img_s_i, out_w_i, in_w_i) in enumerate(
                zip(image_size, seg_prob.shape[2:], window_data.shape[2:])
            ):
                _scale = out_w_i / float(in_w_i)
                if not (img_s_i * _scale).is_integer():
                    warnings.warn(
                        f"For spatial axis: {axis}, output[{ss}] will have non-integer shape. Spatial "
                        f"zoom_scale between output[{ss}] and input is {_scale}. Please pad inputs."
                    )
                zoom_scale.append(_scale)

            if _initialized_ss < ss:  # init. the ss-th buffer at the first iteration
                # construct multi-resolution outputs
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                _initialized_ss += 1

            # resizing the importance_map
            resizer = Resize(spatial_size=seg_prob.shape[2:], mode="nearest", anti_aliasing=False)

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                # zoom roi
                original_idx_zoom = list(original_idx)  # 4D for 2D image, 5D for 3D image
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or (not zoomed_end.is_integer()):
                        warnings.warn(
                            f"For axis-{axis-2} of output[{ss}], the output roi range is not int. "
                            f"Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). "
                            f"Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. "
                            f"Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\n"
                            f"Please change overlap ({overlap}) or roi_size ({roi_size[axis-2]}) for axis-{axis-2}. "
                            "Tips: if overlap*roi_size*zoom_scale is an integer, it usually works."
                        )
                    original_idx_zoom[axis] = slice(int(zoomed_start), int(zoomed_end), None)
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[0].to(compute_dtype)
                # store results and weights
                output_image_list[ss][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]
                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[ss][original_idx_zoom].shape)
                )

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list.pop(0)).to(compute_dtype)

    # remove padding if image_size smaller than roi_size
    for ss, output_i in enumerate(output_image_list):
        if torch.isnan(output_i).any() or torch.isinf(output_i).any():
            warnings.warn("Sliding window inference results contain NaN or Inf.")

        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)
        ]

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(
                int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]

    if dict_key is not None:  # if output of predictor is a dict
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)  # type: ignore
    return final_output[0] if is_tensor_output else final_output  # type: ignore

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

def check_data(dataset_check):
    img = dataset_check[0]["image"]
    label = dataset_check[0]["label"]
    print(dataset_check[0]["name"])
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    print(torch.unique(label[0, :, :, 150]))
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, 150].detach().cpu(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, 150].detach().cpu())
    plt.show()