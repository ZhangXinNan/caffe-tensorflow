./convert.py /Users/zhangxin/github/caffe/models/bvlc_googlenet/deploy.prototxt \
    --caffemodel /Users/zhangxin/github/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel \
    --data-output-path=bvlc_googlenet.npy


./convert.py /Users/zhangxin/github/caffe/models/bvlc_googlenet/deploy.prototxt \
    --code-output-path=bvlc_googlenet.py


./classify.py  models/googlenet.npy /Users/zhangxin/github/caffe/examples/images/cat.jpg



