# googlenet
./convert.py /Users/zhangxin/github/caffe/models/bvlc_googlenet/deploy.prototxt \
    --caffemodel /Users/zhangxin/github/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel \
    --data-output-path=googlenet.npy


./convert.py /Users/zhangxin/github/caffe/models/bvlc_googlenet/deploy.prototxt \
    --code-output-path=googlenet.py


./classify.py  models/googlenet.npy /Users/zhangxin/github/caffe/examples/images/cat.jpg


# 存在问题
## 去掉最后一个softmax，googlenet.py
```
        (self.feed('inception_5b_1x1',
                   'inception_5b_3x3',
                   'inception_5b_5x5',
                   'inception_5b_pool_proj')
             .concat(3, name='inception_5b_output')
             .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5_7x7_s1')
             .fc(1000, relu=False, name='loss3_classifier')
        )#.softmax(name='prob'))
```
## dataset.py中有一处错误
```
img = tf.image.resize_images(img, new_shape[0], new_shape[1])
改为：
img = tf.image.resize_images(img, (new_shape[0], new_shape[1]))
```

## 程序移到其他地方时要注意与kaffe的路径，在helper.py里设置
```
# Add the kaffe module to the import path
sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), '../../../')))
```