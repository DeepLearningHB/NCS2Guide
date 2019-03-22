# Well-driven model's structure and conversion process
```python
def model(x, keep_drop=keep_drop):
    #x = tf.cast(x, tf.float32)
    x_norm = x / 255.0
    net = slim.conv2d(x_norm, 32, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 64 32

    net = slim.conv2d(net, 64, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 32 16

    net = slim.conv2d(net, 128, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 16 8

    net = slim.conv2d(net, 256, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 8 4
    net = tf.nn.dropout(net, keep_prob=keep_drop)

    net = slim.conv2d(net, 512, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 4 2

    net = slim.conv2d(net, 1024, kernel_size=(3, 3))

    net = slim.flatten(net)
    net = tf.nn.dropout(net, keep_prob=keep_drop)

    net_t = slim.fully_connected(net, 8)
    net_t_soft = tf.nn.softmax(net_t)

    net_c = slim.fully_connected(net, 2)
    net_c_soft = tf.nn.softmax(net_c, name='hypothesis')
    return net_t, net_c, net_t_soft, net_c_soft
```
## Placeholder, cost function, model optimizer

```python
X = tf.placeholder(tf.float32, [None, train_x.shape[1], train_x.shape[2], train_x.shape[3]], name='input')
Y = tf.placeholder(tf.int64, [None], name='Y')
Y_p = tf.where(Y > 0, tf.ones_like(Y), tf.zeros_like(Y))
# P = tf.placeholder(tf.float32, name='Dropout')
y_hat_t, y_hat_o, _,_ = model(X)

with tf.name_scope('Optimizer'):
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat_t, labels=Y))
    # if cost == Nan.
    # case 1: divide 0
    # case 2: logN = 0
    cost_o = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat_o, labels=Y_p))

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.015).minimize(cost+cost_o)
```

## Check my model using summarize_graph.py in /model_optimizer/mo/utils/
```
leehanbeen@cvpr:/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/mo/utils$ sudo python3 summarize_graph.py
\ --input_model /home/leehanbeen/PycharmProjects/TypeClassifier/inference_graph_type.pb
```

- result
```
1 input(s) detected:
Name: input, type: float32, shape: (-1,64,128,3)
1 output(s) detected:
hypothesis
```
## Model optimizing
```
sudo python3 mo_tf.py
\ --input_model /home/leehanbeen/PycharmProjects/TypeClassifier/inference_graph_type.pb
\ --input_shape "[1, 64, 128, 3]"
\ --input "input"
\ --data_type FP16
```
- result
```
Model Optimizer arguments:
Common parameters:
        - Path to the Input Model:      /home/leehanbeen/PycharmProjects/TypeClassifier/inference_graph_type.pb
        - Path for generated IR:        /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/.
        - IR output name:       inference_graph_type
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         input
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         [1, 64, 128, 3]
        - Mean values:  Not specified
        - Scale values:         Not specified
        - Scale factor:         Not specified
        - Precision of IR:      FP16
        - Enable fusing:        True
        - Enable grouped convolutions fusing:   True
        - Move mean values to preprocess section:       False
        - Reverse input channels:       False
TensorFlow specific parameters:
        - Input model in text protobuf format:  False
        - Offload unsupported operations:       False
        - Path to model dump for TensorBoard:   None
        - List of shared libraries with TensorFlow custom layers implementation:        None
        - Update the configuration file with input/output node names:   None
        - Use configuration file used to generate the model with Object Detection API:  None
        - Operations to offload:        None
        - Patterns to offload:  None
        - Use the config file:  None
Model Optimizer version:        1.5.12.49d067a0
[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/./inference_graph_type.xml
[ SUCCESS ] BIN file: /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/./inference_graph_type.bin
[ SUCCESS ] Total execution time: 2.23 seconds.
```
## Source code in Raspberry PI
```python
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2
import time


BIN_PATH = '/home/pi/Downloads/inference_graph_type.bin'
XML_PATH = '/home/pi/Downloads/inference_graph_type.xml'

IMAGE_PATH = '/home/pi/Downloads/plate(114).jpg_1.jpg'

print(cv2.__version__)
s_time = time.time()

net = IENetwork(XML_PATH, BIN_PATH)
plugin = IEPlugin(device='MYRIAD')
exec_nt = plugin.load(net)

net_load_time = time.time()

frame = cv2.imread(IMAGE_PATH)
frame = cv2.resize(frame, (128, 64))
blob = cv2.dnn.blobFromImage(frame, size=(128, 64), ddepth=cv2.CV_32F)

preprocess_time = time.time()


res = exec_nt.infer({'input': blob})

inference_time = time.time()

print(res['hypothesis'])
print(np.argmax(res['hypothesis']))
print(np.max(res['hypothesis']))

print("---------------------")
print("Network Loading Time: %s" % (net_load_time - s_time))
print("Image Preprocessing Time: %s" % (preprocess_time - net_load_time))
print("Inference Time: %s" % (inference_time - preprocess_time))
```
- result
```
4.0.1-openvino
[[0.15551758 0.70458984 0.1303711 0.00165462 0.001620217 0.00432587
0.00094889 0.00092173]]
1
0.70458984
---------------------
Network Loading Time: 0.045161008834839
Image Preprocessing Time: 0.05246901512145996
Inference Time: 0.01130819320678711
```
