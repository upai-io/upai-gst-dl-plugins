# upai-gst-dl-plugins
Guide &amp; Examples to create deeplearning gstreamer plugins and use them in your pipeline


# Introduction
Thanks to the work done by [@jackersson](https://github.com/jackersson).
In this repository we have taken inspirations from:

1. [gstreamer-python](https://github.com/jackersson/gstreamer-python)
2. [gst-plugins-tf](https://github.com/jackersson/gst-plugins-tf)
   


# Installation
We have currently tested our code on `Ubuntu 18.04`. You can also refer the official [installation](https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c#install-gstreamer-on-ubuntu-or-debian) document for your linux flavor.

## Packages Required

```
sudo apt install cmake m4 git build-essential

sudo apt install libssl-dev libcurl4-openssl-dev liblog4cplus-dev
```



## Gstreamer Installation
```
sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-libav
```

## Gstreamer Python Binding Installation
Please follow the [script](https://github.com/jackersson/gstreamer-python/blob/master/build-gst-python.sh) provided by [@jackersson](https://github.com/jackersson) for installation of gstreamer binding.


