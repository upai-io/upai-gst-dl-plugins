"""
Usage
    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
    export GST_DEBUG=python:4

    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
        gst_tf_detection config=data/tf_object_api_cfg.yml ! videoconvert ! autovideosink
"""

import os
import logging
import typing as typ
import json

import cv2
import numpy as np

import mxnet as mx
from mxnet import nd
from gluoncv import data, model_zoo

from gstreamer import Gst, GObject, GstBase, GstVideo
import gstreamer.utils as utils
from gstreamer.gst_objects_info_meta import gst_meta_write


def _get_log_level() -> int:
    return int(os.getenv("GST_PYTHON_LOG_LEVEL", logging.DEBUG / 10)) * 10


log = logging.getLogger("gst_python")
log.setLevel(_get_log_level())


class GluonCVModelBaseClass(object):
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "faster_rcnn_resnet50_v1b_voc") # Default Model is Faster Rcnn
        self.pre_trained = kwargs.get('pre_trained', True)
        self.class_list = kwargs.get('class_list', None)
        self.reuse_weights = kwargs.get('reuse_weights', None)
        self.params = kwargs.get('params', None)
        self.reset_class = kwargs.get('reset_class', False)
        self.n_gpu = mx.context.num_gpus()
        self.ctx = (
            [mx.gpu(0), mx.gpu(1)]
            if self.n_gpu >= 2
            else [mx.gpu()]
            if self.n_gpu == 1
            else [mx.cpu(), mx.cpu()]
        )
        self.model = self._load_model()

    def _load_model(self):
        net = model_zoo.get_model(
            self.model_name, pretrained=self.pre_trained, ctx=self.ctx
        )
        if self.params:
            logging.info(f"Loading params {self.params}")
            net.load_parameters(self.params)

        if self.reset_class:
            logging.info("Resetting class for custom classes")
            logging.info(f"Custom Class List {self.class_list}")
            if not self.class_list:
                raise Exception("No class list provided")

            try:
                net.reset_class(self.class_list, reuse_weights=self.reuse_weights)
                logging.info("Successfully loaded custom weights")
            except Exception:
                net.reset_class(self.class_list, reuse_weights=self.reuse_weights)
                logging.info("Successfully loaded custom weights")
        return net

    def resize_bboxes(self, bbox, in_size, out_size):
        if not len(in_size) == 2:
            raise ValueError(
                "in_size requires length 2 tuple, given {}".format(len(in_size))
            )
        if not len(out_size) == 2:
            raise ValueError(
                "out_size requires length 2 tuple, given {}".format(len(out_size))
            )

        bbox = bbox.copy().astype(float)
        x_scale = out_size[0] / in_size[0]
        y_scale = out_size[1] / in_size[1]
        bbox[:, 1] = y_scale * bbox[:, 1]
        bbox[:, 3] = y_scale * bbox[:, 3]
        bbox[:, 0] = x_scale * bbox[:, 0]
        bbox[:, 2] = x_scale * bbox[:, 2]
        return bbox

    def get_output(  # noqa:C901
        self,
        img,
        bboxes,
        scores=None,
        labels=None,
        thresh=0.5,
        class_names=None,
        absolute_coordinates=True,
    ):

        if labels is not None and not len(bboxes) == len(labels):
            raise ValueError(
                "The length of labels and bboxes mismatch, {} vs {}".format(
                    len(labels), len(bboxes)
                )
            )
        if scores is not None and not len(bboxes) == len(scores):
            raise ValueError(
                "The length of scores and bboxes mismatch, {} vs {}".format(
                    len(scores), len(bboxes)
                )
            )

        if len(bboxes) < 1:
            return None
        output = []
        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()

        if not absolute_coordinates:
            # convert to absolute coordinates using image shape
            height = img.shape[0]
            width = img.shape[1]
            bboxes[:, (0, 2)] *= width
            bboxes[:, (1, 3)] *= height

        bboxes = self.resize_bboxes(
            bboxes,
            (self.trans_width, self.trans_height),
            (self.orig_width, self.orig_height),
        )

        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < thresh:
                continue
            if labels is not None and labels.flat[i] < 0:
                continue
            cls_id = int(labels.flat[i]) if labels is not None else -1
            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ""
            score = "{:.3f}".format(scores.flat[i]) if scores is not None else ""
            if class_name or score:
                xmin, ymin, xmax, ymax = [int(x) for x in bbox]
                output.append(
                    {
                        "class_name": class_name,
                        "confidence": score,
                        "bounding_box": [xmin, ymin, xmax, ymax],
                        "bbox": {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                        },
                    }
                )
        return output

    def _pre_process(self, model_type, image):
        self.orig_img = nd.array(image)
        self.orig_height = image.shape[0]
        self.orig_width = image.shape[1]
        if model_type == "yolo":
            x, img = data.transforms.presets.yolo.transform_test(
                self.orig_img, short=512
            )

        if model_type == "faster_rcnn":
            x, img = data.transforms.presets.rcnn.transform_test(
                self.orig_img, short=512
            )
        self.trans_height = img.shape[0]
        self.trans_width = img.shape[1]
        if self.n_gpu >= 1:
            x = x.copyto(mx.gpu(0))
        return x, img

    def predict(self, architecture, image, confidence):
        logging.info("Image received for detection")
        x, img = self._pre_process(architecture, image)
        logging.info("pre processing for image completed")
        class_ids, scores, bounding_boxes = self.model(x)
        logging.info("prediction completed")
        response = self.get_output(
            img,
            bounding_boxes[0],
            scores[0],
            class_ids[0],
            class_names=self.model.classes,
            thresh=confidence,
        )
        logging.info("output generated")
        logging.info(f"Response from prediction {response}")
        if not response:
            logging.debug("No Predictions")
            return []

        return response


def load_config(filename: str) -> dict:
    if not os.path.isfile(filename):
        raise ValueError(f"Invalid filename {filename}")

    with open(filename, "r") as config_file:
        try:
            config = json.loads(config_file.read())
            return config
        except Exception as exc:
            raise OSError(f"Parsing error. Filename: {filename}")


def from_config_file(filename: str) -> GluonCVModelBaseClass:
    """
    :param filename: filename to model config
    """
    return GluonCVModelBaseClass(**load_config(filename))


class GstBasePluginClass(GstBase.BaseTransform):

    # Metadata Explanation:
    # http://lifestyletransfer.com/how-to-create-simple-blurfilter-with-gstreamer-in-python-using-opencv/

    GST_PLUGIN_NAME = "gst_mxnet_detection"

    __gstmetadata__ = ("Name", "Transform", "Description", "Author")

    _srctemplate = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw,format=RGB"),
    )

    _sinktemplate = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw,format=RGB"),
    )

    __gsttemplates__ = (_srctemplate, _sinktemplate)

    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        "config": (
            str,
            "Path to config file",
            "Contains path to config *.yml supported by Gluoncv",
            None,  # default
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()

        self.model = None
        self.config = None

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        return True

    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:

        if self.model is None:
            Gst.error(
                f"No model speficied for {self}"
            )
            return Gst.FlowReturn.ERROR

        try:
            # Convert Gst.Buffer to np.ndarray
            image = utils.gst_buffer_with_caps_to_ndarray(
                buffer, self.sinkpad.get_current_caps()
            )

            # model inference
            objects = self.model.predict("faster_rcnn", image, 0.80)

            Gst.debug(
                f"Frame id ({buffer.pts // buffer.duration}). Detected {str(objects)}"
            )

            # write objects to as Gst.Buffer's metadata
            # Explained: http://lifestyletransfer.com/how-to-add-metadata-to-gstreamer-buffer-in-python/
            gst_meta_write(buffer, objects)
        except Exception as err:
            logging.error("Error %s: %s", self, err)

        return Gst.FlowReturn.OK

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "model":
            return self.model
        if prop.name == "config":
            return self.config
        else:
            raise AttributeError("Unknown property %s" % prop.name)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == "model":
            self._do_set_model(value)
        elif prop.name == "config":
            self._do_set_model(from_config_file(value))
            self.config = value
            Gst.info(f"Model's config updated from {self.config}")
        else:
            raise AttributeError("Unknown property %s" % prop.name)

    def _do_set_model(self, model: GluonCVModelBaseClass):
        self.model = model

    def __exit__(self, exc_type, exc_val, exc_tb):

        Gst.info(f"All frame {self}")


# Required for registering plugin dynamically
# Explained: http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstBasePluginClass)
__gstelementfactory__ = (
    GstBasePluginClass.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    GstBasePluginClass,
)
