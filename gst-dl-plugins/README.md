# Python plugins

## How to test Gluoncv Base Plugin?

1. Install the libraries in the python environment
```python
pip install mxnet gluoncv opencv-python
```

2. Create a directory structure for the plugin and keep the plugin inside a `python` folder.
> For example - /home/ubuntu/<sample_plugin_directory>/python/gst_gluoncv_base_plugin.py



3. Set the `GST_PLUGIN_PATH` environment variable
> export GST_PLUGIN_PATH=/home/usr/lib/gstreamer-1.0/:/home/ubuntu/sample_plugin_directory/python/gst_gluoncv_base_plugin.py

`To check if gstreamer is able to detect the plugin`

> gst-inspect-1.0 <plugin_name>

`gst-launch-1.0 gst_gluoncv`


1. Create a sample JSON config for the plugin
   
`gluon_plugin_config.json`
```json
{
      "model_name": "faster_rcnn_resnet50_v1b_voc",
      "pre_trained": true,
      "class_list": null,
      "reuse_weights": null,
      "params": null,
      "reset_class": false
}
```


## Sample Gstreamer Pipes

1. ``