# Neural Compute Stick 2 환경 구성 및 동작
## OS
- Ubuntu LTS 16.04
- Rasbian Stretch

## Raspberry PI 3 환경 설정
라즈베리파이에 라즈비안 스트레치가 설치되어있다고 가정하고 아래 내용을 진행한다.
[이 곳](http://download.01.org/openvinotoolkit/2018_R5/packages/l_openvino_toolkit_ie_p_2018.5.445.tgz)을 클릭하여 Raspberry PI 용 OpenVINO를 다운로드 받는다.
이 설치 패키지에는 OpenCV 4.0버전과 추론 엔진 및 샘플 애플리케이션이 포함되어 있다.

기본 설치 경로는 ```~/Downloads```이다.

설치 후 압축 파일의 압축을 푼다.
```
tar -xf <l_openvino_toolkit_ie_p_<version>.tgz
```

설치된 폴더의 절대경로로 ```setupvars.sh``` 를 바꾸어 스크립트를 수정한다.

이 과정을 완료하면 OpenVINO를 사용할 준비가 완료 되었고, 환경 변수 설정을 해야한다.

임시로 환경 변수를 설정하기 위해서
```
source /Downloads/inference_engine_vpu_arm/bin/setupvars.sh
```
를 입력하지만 재실행 후에도 유지를 하기 위해서 ```.bashrc```파일을 변경해줘야한다.
최상위 디렉토리로 이동 후 ```vi .bashrc```를 이용해 편집한다.

` bashrc ` 파일 최 하단에 환경 변수 설정 코드를 넣어주면 재실행 후에도 유지가 된다.

```
[setupvars.sh] OpenVINO environment initialized
```
메시지가 터미널을 실행 시 나타나면 성공적으로 설정된 것이다.


```
sudo usermod -a -G users "$(whoami)"
```
현재 리눅스 사용자를 users 그룹에 추가한다, 로그아웃 후 로그인 시 적용된다.
Inference를 하기 위해서 Neural Compute Stick 2를 설치(?)한다.
```
sh inference_engine_vpu_arm/install_dependencies/install_NCS_udev_rules.sh
```
이 과정을 마치면 Neural Compute Stick 2를 라즈베리파이에서 구동할 준비가 완료된다.

## 데스크탑(리눅스) 환경 구성
설치 환경은 Ubuntu 16.04 LTS 이다.
[이 곳](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux)을 클릭하여 OpenVINO 인텔 배포판을 다운로드한다.

기본 설치 경로는 `cd ~/Downloads`에 설치된다.
```
tar -zxf l_openvino_toolkit_p_<version>.tgz
```
압축 파일을 푼다.
```
cd l_openvino_toolkit_p_<version>
```
```
sudo -E ./install_cv_sdk_dependencies.sh
```
실행파일을 실행하면 종속적인 외부 소프트웨어를 자동으로 다운로드하고 설치한다.
OpenVINO 핵심 구성 요소를 설치하기 위해 실행 파일을 실행한다.
```
sudo ./install_GUI.sh
```
```
sudo ./install.sh
```
이 두가지 명령의 차이점은 설치를 GUI 환경으로 하느냐, 텍스트 환경에서 하느냐의 차이가 유일하다.

이 과정을 완료하면  `cd /opt/intel/openvino` 경로에 `openvino`파일이 생긴다고 공식 문서에는 설명이 되어 있지만, `/opt/intel/computer_vision_sdk`라고 설치가 되었다.

OpenVINO 애플리케이션을 컴파일하고 실행하려면 환경 변수들을 업데이트 해야한다.
```
source /opt/intel/computer_vision_sdk/bin/setupvars.sh
```
를 통해 설정할 수 있고, 이 소스 역시 `bashrc`파일에 추가하면 재실행 시에도 환경이 유지된다.


## Model Optimizer
데스크탑에서 설계한 모델을 라즈베리파이에서 구동하기 위해서 일련의 변환 작업이 필요하다.
여기서는 모델의 체크포인트를 저장한 `.ckpt`파일이 있다고 가정한다.
먼저 이 `.ckpt` 파일을 `.pb` 파일로 변환해 주는 작업이 필요하다.
이 변환하는 방법은 여러가지가 있지만, `tensorflow`에서 제공하는 함수를 이용하기로 한다.

```python
from tensorflow.python.framework import graph_io
...
...
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["hypothesis"])
graph_io.write_graph(frozen, './saved/model/', 'inference_graph_char.pb', as_text=False)

```
`tf.graph_util.convert_variables_to_constants` 함수에서 변수들을 동결시킨다. 마지막 인자로 들어간 `["hypothesis"]`는 출력 값의 `name_scope`를 지정해 준 것이다.
`graph_io.write_graph` 함수의 인자로 동결된 값, 저장 장소, 저장 파일명, `as_text` 가 들어가는데 `as_text`는 `False`로 주었다. (용도는 잘 모르겠다.)

이 과정에서 오류가 발생한다면 아마 `name` 지정 문제가 가장 크리라 생각된다.

이 `.pb` 파일을 다시한번 변환을 해주어야 라즈베리파이에서 사용이 가능하다.
`.xml`파일과 `.bin`파일로 다시 변환을 해줘야하는데,
`.xml`파일은 그래프의 구조를 저장한 것이고, `.bin`파일은 그래프에서 사용하는 `weight`를 저장한 파일이다.

이 변환 과정을 `Intermediate Representation (IR)`라고 하고, 직역하면 중간 표현으로 변환한다는 뜻이다.
다운로드 받은 OpenVINO Toolkit의 `model_optimizer`를 여기서 사용한다.

먼저 `model_optimizer`의 경로로 찾아 들어간다.
```
cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer
```
```
extensions             mo_onnx.py              requirements_onnx.txt
install_prerequisites  mo.py                   requirements_tf.txt
mo                     mo_tf.py                requirements.txt
mo_caffe.py            requirements_caffe.txt  tf_call_ie_layer
mo_kaldi.py            requirements_kaldi.txt  version.txt
mo_mxnet.py            requirements_mxnet.txt
```
`model_optimizer`파일의 내부이다.
`mo.py`를 이용한 변환 전에 설치해야 할 것이 있다.
`cd install_prerequisites`로 들어가서  `./install_prerequisites.sh`로 구성 요소들을 설치한다.
`tensorflow`만 이용 시 `./install_prerequisites_tf.sh`만 이용해도 무방할 것으로 생각된다.

권장되는 사항은 `/mo/utils/summarize_graph.py`를 이용해 그래프의 구조를 재확인 하는 것이다.
```
sudo python3 summarize_graph.py --input_model /home/leehanbeen/PycharmProjects/TypeClassifier/inference_graph_type.pb
```
```
1 input(s) detected:
Name: input, type: float32, shape: (-1,64,128,3)
1 output(s) detected:
hypothesis
```
위에서 나오는 대로 IR로 변환 시 입력 매개변수에 넣어줘야한다.
만약 여기서 dropout같이 의도하지 않은 입력이 출력되면 모델을 수정을 해야한다.
dropout 같은 것에 name을 지정해주면 이를 입력으로 인식하는 듯하다.

다시 `model_optimizer` 폴더로 들어와서 변환을 수행해준다.
```
sudo python3 mo_tf.py --input_model /home/leehanbeen/PycharmProjects/TypeClassifier/inference_graph_type.pb --input_shape "[1, 64, 128, 3]" --input "input"
```
`--input_model` 매개변수는 `.pb`파일의 경로, `--input_shape` 매개변수는 input의 shape을 넣어줘야하는데,
여기서 모델 훈련 시 `placeholder`는 `None`을 넣었지만, 변환 시에는 `1`로 넣어줘야 한다.
`--input` 매개변수는 입력 placeholder의 `name`을 넣어주면 된다.
기타 매개변수 사용은 [이 곳](https://software.intel.com/en-us/articles/OpenVINO-Using-Caffe#using-framework-agnostic-conv-param)을 참조하면 된다.

또 주의할 것이 흑백 이미지를 입력으로 받을 시 placeholder를  `[None, H, W]`와 같이 `Channel`을 지정하지
않는 경우가 있는데, 이럴 경우 변환이 안되는 것 같다. 따라서 모델 훈련 시 `[None, H, W, C]` 형태로
훈련하는 것이 나으리라 생각된다.

```
[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/./inference_graph_type.xml
[ SUCCESS ] BIN file: /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/./inference_graph_type.bin
[ SUCCESS ] Total execution time: 2.41 seconds.
```

이런 메시지가 뜨면 성공적으로 변환된 것이고, 저장되는 기본 디렉토리는 `/opt/intel/computer_vision_sdk/deployment_tools/model_optimizer`이다.

만약 변환 시 오류가 발생한다면 [이 곳](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer#FAQ)을 참고하면 된다.

이제 데스크탑에서의 작업은 모두 완료되었다.
여기서 생성한 `.xml`, `.bin` 파일을 USB등을 통해 `Raspberry PI`에 옮겨서 소스를 작성할 것이다.

## Raspberry PI를 이용한 Inference

위 과정이 모두 성공적으로 종료되면 나머지는 간단하다.

```python
import tensorflow as tf
import cv2
import numpy as np
BIN_PATH = '/home/pi/Downloads/inference_graph_type.bin'
XML_PATH = '/home/pi/Downloads/inference_graph_type.xml'
IMAGE_PATH = '/home/pi/Downloads/plate(110).jpg_2.jpg' #naming miss.. :(
```
여기서 `OpenCV`의 버전은 4.0.0 이상이여야 한다.
```python
net = cv2.dnn.readNet(XML_PATH, BIN_PATH)
net.setPreferableTarget(cv2.Dnn.DNN_TARGET_MYRIAD)
```
`readNet`의 매개변수로 `.xml`과 `.bin`파일의 경로를 잡아준다.
`setPreferableTarget`으로 기기를 인식하는 것으로 보인다.

```python
frame = cv2.imread(IMAGE_PATH)
frame = cv2.resize(frame, (128, 64))
blob = cv2.dnn.blobFromImage(frame, size=(128, 64), ddepth=cv2.CV_8U)
```
입력에 맞게 이미지를 전처리를 수행해준다.
네트워크에 들어가기 위해서 이미지를 `blob`이라는 형태로 변경해주는 것으로 보인다.

```python
net.setInput(blob)
out = net.forward()
```
입력을 넣고 출력을 얻는다. `net.forward()` 메소드로 간단하게 결과를 얻을 수 있다.

전체 소스코드는 아래와 같다.
```python
import tensorflow as tf
import cv2
import numpy as np
BIN_PATH = '/home/pi/Downloads/inference_graph_type.bin'
XML_PATH = '/home/pi/Downloads/inference_graph_type.xml'
IMAGE_PATH = '/home/pi/Downloads/plate(110).jpg_2.jpg' #naming miss.. :(
net = cv2.dnn.readNet(XML_PATH, BIN_PATH)
net.setPreferableTarget(cv2.Dnn.DNN_TARGET_MYRIAD)
frame = cv2.imread(IMAGE_PATH)
frame = cv2.resize(frame, (128, 64))
blob = cv2.dnn.blobFromImage(frame, size=(128, 64), ddepth=cv2.CV_8U)
net.setInput(blob)
out = net.forward()
out = out.reshape(-1)
print(out)
print(np.max(out))
print(np.argmax(out))
```
실행 결과
```
[0.0128479 0.2097168 0.76416016 0.00606918 0.00246811 0.00198746 0.00129604 0.00117588]
0.76416016
2
```
의도한 출력이 생성되었다.
