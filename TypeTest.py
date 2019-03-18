import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import cv2
import shutil
from TypeClassifier import data_read
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.framework import graph_io
from utils import coldGraph
file = open("test.txt", "w")
file_train = open("train.txt", "w")
URL = '/home/leehanbeen/PycharmProjects/TypeClassifier/CroppedImage/'
batch_size = 2000
def model(x, keep_drop=1.0):
    #x = tf.cast(x, tf.float32)
    x_norm = tf.divide(x, 255.0)
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
    net_t_soft = tf.nn.softmax(net_t, name='hypothesis')
    # 이 부분에서 .pb파일로 변환 시 들어갈 output의 이름을 정해주었음.
    
    net_c = slim.fully_connected(net, 2)
    net_c_soft = tf.nn.softmax(net_c)
    return net_t, net_c, net_t_soft, net_c_soft


if __name__ == "__main__":
    data_x, data_y, img_list, path_list = data_read(URL)
    print(data_x.shape, data_y.shape)

    train_x, test_x, train_y, test_y, img_list_train, img_list_test, path_list_train, path_list_test = train_test_split(data_x, data_y, img_list, path_list,  test_size=0.3, shuffle=True, random_state=42)
    train_y_o = np.where(train_y > 0, 1, 0)
    test_y_o = np.where(test_y >0, 1, 0)

    print(len(img_list_train))
    print(len(img_list_test))

    print(len(train_y), len(train_x), len(test_x), len(test_y))
   # print(type(train_t))
    print(train_x.shape[1], train_x.shape[2])
    X = tf.placeholder(tf.float32, [None, train_x.shape[1], train_x.shape[2], train_x.shape[3]], name='input')
    Y = tf.placeholder(tf.int64, [None], name='Y')
    Y_p = tf.where(Y > 0, tf.ones_like(Y), tf.zeros_like(Y))
    #P = tf.placeholder(tf.float32, name='Dropout')

    y_hat_t, y_hat_o, y_hat_t_soft, y_hat_o_soft = model(X)

    argmax = tf.argmax(y_hat_t, -1)
    correct_prediction = tf.equal(argmax, Y)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    argmax_o = tf.argmax(y_hat_o, -1)
    correct_prediction_o = tf.equal(argmax_o, Y_p)
    acc_o = tf.reduce_mean(tf.cast(correct_prediction_o, tf.float32))


    feed_dict = {X: train_x, Y: train_y, Y_p: train_y_o}
    feed_dict1 = {X: test_x, Y: test_y, Y_p: test_y_o}


    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')
    else:
        print("Model load failed.")
        exit()

    total_batch = int(len(test_x) / batch_size)

    avg_acc_t = 0
    avg_acc_o = 0

    avg_acc_train_t = 0
    avg_acc_train_o = 0
    y_hat_t_list = []
    y_hat_o_list = []
    y_hat_t_soft_list = []
    y_hat_o_soft_list = []

    y_hat_train_t_list = []
    y_hat_train_o_list = []
    y_hat_train_t_soft_list = []
    y_hat_train_o_soft_list = []

    total_batch_train = int(len(train_x) / batch_size)

    for i in range(total_batch_train):
        start = ((i) * batch_size)
        end = ((i+1) * batch_size)

        batch_train_xs = train_x[start:end]
        batch_train_ys = train_y[start:end]
        batch_train_ys_o = train_y_o[start:end]

        feed_dict_train = {X: batch_train_xs, Y: batch_train_ys, Y_p: batch_train_ys_o}
        _y_hats_train, _y_hats_o_train, acc_t_train, acc_t_o_train, argmax_t_train, argmax_t_o_train,\
            soft_t, soft_o= sess.run([y_hat_t, y_hat_o, acc, acc_o, argmax, argmax_o, y_hat_t_soft, y_hat_o_soft], feed_dict=feed_dict_train)

        avg_acc_train_t += acc_t_train/total_batch_train
        avg_acc_train_o += acc_t_o_train/total_batch_train

        y_hat_train_t_list.extend(argmax_t_train)
        y_hat_train_o_list.extend(argmax_t_o_train)
        y_hat_train_t_soft_list.extend(np.max(soft_t, axis=1))
        y_hat_train_o_soft_list.extend(np.max(soft_o, axis=1))


    for i in range(total_batch):
        start = ((i) * batch_size)
        end = ((i+1) * batch_size)
        batch_xs = test_x[start:end]
        batch_ys = test_y[start:end]
        batch_ys_o = test_y_o[start:end]

        feed_dict1 = {X: batch_xs, Y: batch_ys, Y_p:batch_ys_o}
        _y_hats, _y_hats_o, acc_t, acc_t_o, argmax_t, argmax_t_o,\
            soft_t_t, soft_t_o= sess.run([y_hat_t, y_hat_o, acc, acc_o, argmax, argmax_o, y_hat_t_soft, y_hat_o_soft], feed_dict=feed_dict1)

        avg_acc_t += acc_t/total_batch
        avg_acc_o += acc_t_o/total_batch

        y_hat_t_list.extend(argmax_t)
        y_hat_o_list.extend(argmax_t_o)
        y_hat_t_soft_list.extend(np.max(soft_t_t, axis=1))
        y_hat_o_soft_list.extend(np.max(soft_t_o, axis=1))

    print(len(y_hat_t_list))
    print(len(y_hat_t_soft_list))
    if not os.path.exists('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage'):
        os.makedirs('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage')
    if not os.path.exists('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage/Type'):
        os.makedirs('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage/Type')
    if not os.path.exists('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage/Object'):
        os.makedirs('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage/Object')

    for i in range(len(test_y[:len(y_hat_t_list)])):
        if test_y[i] != y_hat_t_list[i]:
            file.write(img_list_test[i] +" TypeError. True: "+str(test_y[i])+ " Prediction: "+str(y_hat_t_list[i])+" "+str(y_hat_t_soft_list[i])+"\n")
            asd = img_list_test[i].split('.')[0]+"_"+str(test_y[i])+"_"+str(y_hat_t_list[i])+"_"+str(y_hat_t_soft_list[i])+"_"+str(y_hat_t_soft_list[i])+'.jpg'
            shutil.copy(os.path.join(path_list_test[i],img_list_test[i]), os.path.join('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage/Type', asd))

    file.write("\n-----------------------------------------\n")

    for i in range(len(test_y_o[:len(y_hat_o_list)])):
        if test_y_o[i] != y_hat_o_list[i]:
            file.write(img_list_test[i] +" Objectness Error. True: "+str(test_y_o[i])+" Prediction: "+str(y_hat_o_list[i])+" "+str(y_hat_o_soft_list[i])+"\n")
            asd = img_list_test[i].split('.')[0]+"_"+str(test_y_o[i])+"_"+str(y_hat_o_list[i])+"_"+str(y_hat_o_soft_list[i])+'.jpg'
            shutil.copy(os.path.join(path_list_test[i],img_list_test[i]), os.path.join('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage/Object', asd))

    for i in range(len(train_y[:len(y_hat_train_t_list)])):
        if train_y[i] != y_hat_train_t_list[i]:
            file_train.write(img_list_train[i] +" TypeError. True: "+str(train_y[i])+" Prediction: "+str(y_hat_train_t_list[i])+" "+str(y_hat_train_t_soft_list[i])+"\n")
            print(img_list_train[i])
            asd = img_list_train[i].split('.')[0]+"_"+str(train_y[i])+"_"+str(y_hat_train_t_list[i])+"_"+str(y_hat_train_t_soft_list[i])+'.jpg'
            shutil.copy(os.path.join(path_list_train[i],img_list_train[i]), os.path.join('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage/Type', asd))
    file_train.write("\n-----------------------------------------\n")

    for i in range(len(train_y_o[:len(y_hat_train_o_list)])):
        if train_y_o[i] != y_hat_train_o_list[i]:
            file_train.write(img_list_train[i] +" Objectness Error. True: "+str(train_y_o[i])+" Prediction: "+str(y_hat_train_o_list[i])+" "+str(y_hat_train_o_soft_list[i])+"\n")
            asd = img_list_train[i].split('.')[0]+"_"+str(train_y_o[i])+"_"+str(y_hat_train_o_list[i])+"_"+str(y_hat_train_o_soft_list[i])+'.jpg'
            shutil.copy(os.path.join(path_list_train[i],img_list_train[i]), os.path.join('/home/leehanbeen/PycharmProjects/TypeClassifier/SavedImage/Object', asd))

    coldGraph(sess, 'model', 'input', 'hypothesis', 'save/Const:hypothesis')
    
    frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["hypothesis"]) #41번째 줄의 hypothesis 가 인자로 들어감.
    graph_io.write_graph(frozen, './', 'inference_graph_type.pb', as_text=False) # 현재 디렉토리에 inference_graph_type.pb 파일 생성.

    print("Test Confusion Matrix")
    print(confusion_matrix(test_y[:len(y_hat_t_list)], y_hat_t_list))
    print(confusion_matrix(test_y_o[:len(y_hat_o_list)], y_hat_o_list))
    print("Test Type Accuracy: %.5f Test Object Accuracy: %.5f" % (avg_acc_t, avg_acc_o))
    print("\n\n")
    print("Train Confusion Matrix")
    print(confusion_matrix(train_y[:len(y_hat_train_t_list)], y_hat_train_t_list))
    print(confusion_matrix(train_y_o[:len(y_hat_train_o_list)], y_hat_train_o_list))
    print("Train Type Accuracy: %.5f Train Object Accuracy: %.5f" % (avg_acc_train_t, avg_acc_train_o))
    file.close()
    file_train.close()


