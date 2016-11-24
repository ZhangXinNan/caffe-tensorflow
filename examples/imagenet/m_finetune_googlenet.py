# Import the converted model's class
import numpy as np
import random
import linecache
import cv2
import argparse
import os
import tensorflow as tf
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))
from  mygooglenet import GoogleNet as MyNet

BATCH_SIZE = 32
LABEL_NUM = 371   #num of classify - 1
NEW_HEIGHT = 224
NEW_WIDTH = 224

def gen_data(source):
    while True:
        myfile = open(source, 'r')  
        indices = range(len(myfile.readlines()))
        print("total num %d\n" %(len(indices)))
        random.shuffle(indices)
        for i in indices:
            line = linecache.getline(source, i)
            content = line.strip()
            #strList = content.split('\t')
            strList = content.split(' ')
            if(len(strList) != 2):                  #num of eachline description list
                print"##format error##"
                continue
            img_file = cv2.imread(strList[0])
            img_label = int(strList[1])
            
            initial_val = 0
            ss = [initial_val for i in range(LABEL_NUM)]
            ss[img_label - 1] = 1   #label start from 0
            image = cv2.resize(img_file,(NEW_HEIGHT,NEW_WIDTH ), interpolation = cv2.INTER_CUBIC)
            label = np.array(ss)

            yield image, label

def gen_data_batch(source):
    data_gen = gen_data(source)
    print"get data_gen\n"
    while True:
        image_batch = []
        label_batch = []
        for _ in range(BATCH_SIZE):
            image, label = next(data_gen)
            image_batch.append(image)
            label_batch.append(label)
        yield np.array(image_batch), np.array(label_batch)
        
def parse_args():
    parser = argparse.ArgumentParser(description='finetune tf model')
    parser.add_argument('-txt_path', dest='txt_abs_path', help='txt containing abs_path and label', 
                        default='./')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #set buff for image data & label
    images = tf.placeholder(tf.float32, [BATCH_SIZE, NEW_HEIGHT, NEW_HEIGHT, 3])
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, LABEL_NUM])
    net = MyNet({'data': images})

    loss3_classifier = net.layers['m_loss3_classifier']   #finetune layers
    pred = tf.nn.softmax(loss3_classifier)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(loss3_classifier, labels), 0)
    opt = tf.train.RMSPropOptimizer(0.001)
    train_op = opt.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        #args = parse_args()

        #txt_file = args.txt_abs_path
        # Load the data
        sess.run(tf.initialize_all_variables())
        net.load('googlenet.npy', sess)
        print "net load ok\n"
        data_gen = gen_data_batch(txt_file)
        for i in range(1000):
            np_images, np_labels = next(data_gen)
            feed = {images: np_images, labels: np_labels}

            np_loss, np_pred, _ = sess.run([loss, pred, train_op], feed_dict=feed)
            if i % 10 == 0:
                print('Iteration: ', i, np_loss)
                savename= os.path.join("m_models", str(i) + "_googlenet.npy")
                print savename
                #saver.save(sess, savename, i)
                saver.save(sess, savename)
