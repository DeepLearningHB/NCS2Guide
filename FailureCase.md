# Error case model

```python
def model(x, keep_prob):
    x_norm = x / 255.0


    net = slim.conv2d(x_norm, 32, kernel_size=(3,3))
    net = slim.max_pool2d(net, (2,2))


    net = slim.conv2d(net, 64, kernel_size=(3,3))
    net = slim.max_pool2d(net, (2,2))

    net = slim.conv2d(net, 128, kernel_size=(3,3))
    net = slim.max_pool2d(net, (2,2))


    net = slim.conv2d(net, 256, kernel_size=(3,3))
    net = slim.max_pool2d(net, (2,2))


    net = slim.flatten(net)
    net = tf.nn.dropout(net, keep_prob)
    logits = slim.fully_connected(net, class_num)


    prob = tf.nn.softmax(logits, name='hypothesis')

    return logits, prob
```
## Placeholder, cost function, model Optimizer

```python
X = tf.placeholder(tf.float32, [None, image_size[1], image_size[0], 3], name='input')
Y = tf.placeholder(tf.int64, [None], name='Y')
#P = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False, name='global_step')

logits, _ = model(X, 0.65)
with tf.name_scope('Optimizer'):
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    tf.summary.scalar('cost' , cost)
```
## Check my model using summarize_graph.py in /model_optimizer/mo/utils/
```
sudo python3 summarize_graph.py
\ --input_model /home/leehanbeen/PycharmProjects/platerecognizechar/saved/model/inference_graph_char.pb
```
- result

```
1 input(s) detected:
Name: input, type: float32, shape: (-1,40,40,3)
1 output(s) detected:
hypothesis
```

## Model optimizing
```
sudo python3 mo_tf.py
\ --input_model /home/leehanbeen/PycharmProjects/saved/model/inference_graph_char.pb
\ --input_shape "[1, 40, 40, 3]"
\ --input "input"
\ --data_type FP16
```
- result
```
Model Optimizer arguments:
Common parameters:
        - Path to the Input Model:      /home/leehanbeen/PycharmProjects/platerecognizechar/saved/model/inference_graph_char.pb
        - Path for generated IR:        /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/.
        - IR output name:       inference_graph_char
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         input
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         [1, 40, 40, 3]
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
[ SUCCESS ] XML file: /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/./inference_graph_char.xml
[ SUCCESS ] BIN file: /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/./inference_graph_char.bin
[ SUCCESS ] Total execution time: 1.61 seconds.
```

## Source code in Raspberry PI
```python
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2
import time


BIN_PATH = '/home/pi/Downloads/inference_graph_char.bin'
XML_PATH = '/home/pi/Downloads/inference_graph_char.xml'

IMAGE_PATH = '/home/pi/Downloads/3.jpg'

print(cv2.__version__)
s_time = time.time()

net = IENetwork(XML_PATH, BIN_PATH)
plugin = IEPlugin(device='MYRIAD')
exec_nt = plugin.load(net)

net_load_time = time.time()

frame = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
frame = cv2.resize(frame, (40, 40))
blob = cv2.dnn.blobFromImage(frame, size=(40, 40), ddepth=cv2.CV_32F)

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

# endless wating....
```
