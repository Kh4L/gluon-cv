"""Transforms for YOLO series."""
from __future__ import absolute_import
import copy
import numpy as np
import mxnet as mx
from mxnet import autograd
from .. import bbox as tbbox
from .. import image as timage
from .. import experimental

from ...batchify import Pad

from ....utils import try_import_dali

dali = try_import_dali()

__all__ = ['transform_test', 'load_test', 'YOLO3DefaultTrainTransform',
           'YOLO3DefaultValTransform', 'YOLO3DALIPipeline']

def transform_test(imgs, short=416, max_size=1024, stride=1, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int, default=416
        Resize image short side to this `short` and keep aspect ratio. Note that yolo network
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our YOLO implementation.
    stride : int, optional, default is 1
        The stride constraint due to precise alignment of bounding box prediction module.
        Image's width and height must be multiples of `stride`. Use `stride = 1` to
        relax this constraint.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.resize_short_within(img, short, max_size, mult_base=stride)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs

def load_test(filenames, short=416, max_size=1024, stride=1, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, default=416
        Resize image short side to this `short` and keep aspect ratio. Note that yolo network
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our YOLO implementation.
    stride : int, optional, default is 1
        The stride constraint due to precise alignment of bounding box prediction module.
        Image's width and height must be multiples of `stride`. Use `stride = 1` to
        relax this constraint.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [mx.image.imread(f) for f in filenames]
    return transform_test(imgs, short, max_size, stride, mean, std)


class YOLO3DefaultTrainTransform(object):
    """Default YOLO training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    net : mxnet.gluon.HybridBlock, optional
        The yolo network.

        .. hint::

            If net is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, width, height, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), mixup=False, **kwargs):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._mixup = mixup
        self._target_generator = None
        if net is None:
            return

        # in case network has reset_ctx to gpu
        self._fake_x = mx.nd.zeros((1, 3, height, width))
        net = copy.deepcopy(net)
        net.collect_params().reset_ctx(None)
        with autograd.train_mode():
            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net(self._fake_x)
        from ....model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
        self._target_generator = YOLOV3PrefetchTargetGenerator(
            num_class=len(net.classes), **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._target_generator is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        if self._mixup:
            gt_mixratio = mx.nd.array(bbox[np.newaxis, :, -1:])
        else:
            gt_mixratio = None
        objectness, center_targets, scale_targets, weights, class_targets = self._target_generator(
            self._fake_x, self._feat_maps, self._anchors, self._offsets,
            gt_bboxes, gt_ids, gt_mixratio)
        return (img, objectness[0], center_targets[0], scale_targets[0], weights[0],
                class_targets[0], gt_bboxes[0])


class YOLO3DefaultValTransform(object):
    """Default YOLO validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype(img.dtype)

class YOLO3DALIPipeline(dali.Pipeline):
    """DALI Pipeline with YOLOv3 training transform.
    Parameters
    ----------
    device_id: int
         DALI pipeline arg - Device id.
    num_workers:
        DALI pipeline arg - Number of CPU workers.
    batch_size:
        Batch size.
    data_shape: int
        Height and width length.
    dataset_reader: Dataset object
        Partial pipeline object, which __call__ function has to return
        (images, bboxes, labels) DALI EdgeReference tuple.
    net : mxnet.gluon.HybridBlock
        The YOLO network.

    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    """
    def __init__(self, num_workers, ctx, batch_size, data_shape,
                 dataset_reader, net=None,  **kwargs):
        super(YOLO3DALIPipeline, self).__init__(
            batch_size=batch_size,
            device_id=ctx.device_id,
            num_threads=num_workers)
        self.dataset_reader = dataset_reader

        # Image
        self.slice = dali.ops.Slice(device="cpu")
        self.resize = dali.ops.Resize(
            device="cpu",
            resize_x=data_shape,
            resize_y=data_shape,
            min_filter=dali.types.DALIInterpType.INTERP_TRIANGULAR)

        output_dtype = dali.types.FLOAT

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            crop=(data_shape, data_shape),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
            output_dtype=output_dtype,
            output_layout=dali.types.NCHW,
            pad_output=False)
        self.twist = dali.ops.ColorTwist(device="gpu")

        # BBox
        self.crop = dali.ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            ltrb=True,
            allow_no_crop=True,
            num_attempts=1)

        self.flip = dali.ops.Flip(device="cpu")
        self.bbflip = dali.ops.BbFlip(device="cpu", ltrb=True)
        self.pad = dali.ops.Pad(device="gpu", padding_value=-1, axis=0)

        # Random variables
        self.flip_coin = dali.ops.CoinFlip(probability=0.5)

        self.rng1 = dali.ops.Uniform(range=[0.5, 1.5])
        self.rng2 = dali.ops.Uniform(range=[0.875, 1.125])
        self.rng3 = dali.ops.Uniform(range=[-0.5, 0.5])

        # Anchor generators
        # in case network has reset_ctx to gpu
        self._batch_size = batch_size
        self._ctx = ctx
        self._fake_x = mx.nd.zeros((1, 3, data_shape, data_shape), ctx=ctx)
        net_copy = copy.deepcopy(net)
        net_copy.collect_params().reset_ctx(ctx)
        with autograd.train_mode():
            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net_copy(self._fake_x)
        from ....model_zoo.yolo.yolo_target import YOLOV3PrefetchTargetGenerator
        self._target_generator = YOLOV3PrefetchTargetGenerator(
            num_class=len(net_copy.classes), **kwargs)
        del net_copy

    def define_graph(self):
        """
        Define the DALI graph.
        """
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        coin_rnd = self.flip_coin()

        images, bboxes, labels = self.dataset_reader()

        crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        images = self.slice(images, crop_begin, crop_size)
        bboxes = self.pad(bboxes.gpu())
        labels = self.pad(labels.gpu())

        return (images, bboxes, labels)

    def get_targets_from_input(self, gt_bboxes, gt_ids):
        assert self._ctx == gt_bboxes.context, "Input and DALI Pipeline context does not match."
        # generate training target so cpu workers can help reduce the workload on gpu
        # gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        # gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])

        # TODO(spanev): Handle MixUp?
        gt_mixratio = None

        objectnesses = []
        center_targets = []
        scale_targets = []
        weights = []
        class_targets = []
        for i in range(self._batch_size):
            bboxes = gt_bboxes[i]
            labels = gt_ids[i]

            valid_labels_mask = (labels != -1)
            bboxes = mx.nd.contrib.boolean_mask(bboxes, valid_labels_mask)
            labels = mx.nd.contrib.boolean_mask(gt_bboxes, valid_labels_mask)

            objectness, center_target, scale_target, weight, class_target = self._target_generator(
                self._fake_x, self._feat_maps, self._anchors, self._offsets,
                bboxes, labels, gt_mixratio)
            objectnesses.append(objectness[0])
            center_targets.append(center_target[0])
            scale_targets.append(scale_target[0])
            weights.append(weight[0])
            class_targets.append(class_target[0])

        # Padding back
        pad = Pad(pad_val=-1)
        objectnesses = pad(objectnesses).as_in_context(self._ctx)
        center_targets = pad(center_targets).as_in_context(self._ctx)
        scale_targets = pad(scale_targets).as_in_context(self._ctx)
        weights = pad(weights).as_in_context(self._ctx)
        class_targets = pad(class_targets).as_in_context(self._ctx)

        return (objectness, center_target, scale_target, weight, class_targets)
