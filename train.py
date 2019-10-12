import tensorflow as tf
import numpy as np
import argparse
import os
import time
from resnet50 import resnet50
from loss import arcface_loss
from tensorflow.python.client import timeline

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)

    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)  # 标准化
    img = tf.multiply(img, 0.0078125)  # 标准化
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def train(tfrecords, batch_size, lr, ckpt_save_dir, epoch, num_classes):
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    tfr = tfrecords
    print('-------------Training Args-----------------')
    print('--tfrecords        :  ', tfr)
    print('--batch_size       :  ', batch_size)
    print('--num_classes      :  ', num_classes)
    print('--ckpt_save_dir    :  ', ckpt_save_dir)
    print('--lr               :  ', lr)
    print('-------------------------------------------')

    dataset = tf.data.TFRecordDataset(tfr)  # 读取tfrecords数据
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()  # (?,112,112,3)

    images = tf.placeholder(tf.float32, [None, 112, 112, 3], name='image_inputs')
    labels = tf.placeholder(tf.int64, [None, ], name='labels_inputs')

    emb = resnet50(images, is_training=True)

    logit = arcface_loss(embedding=emb, labels=labels, w_init=w_init_method, out_num=num_classes)  # (?,85742)
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

    p = int(512.0 / batch_size)  # 512/64 = 8
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    lr_steps = [p * val for val in [40000, 60000, 80000]]  # [320000, 480000, 640000]
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=lr,
                                     name='lr_schedule')  # (x, [320000, 480000, 640000], [0.001, 0.0005, 0.0003, 0.0001]) 学习率迭代形参x指的是global_step，其实就是迭代次数，boundaries一个列表，内容指的是迭代次数所在的区间，values是个列表，存放在不同区间该使用的学习率的值
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)  # 动量优化器

    grads = opt.compute_gradients(inference_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):  # apply_gradients只会在update_ops运行之后运行
        train_op = opt.apply_gradients(grads, global_step=global_step)  # 貌似是BN里面存储mean和std用的
    pred = tf.nn.softmax(logit)  # (?,85742)，softmax概率结果
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))  # 计算正确率

    saver = tf.train.Saver(max_to_keep=3)
    counter = 0
    with tf.Session(config=gpuConfig) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # saver.restore(sess, '/data/ChuyuanXiong/backup/face_real330_ckpt/Face_vox_iter_271800.ckpt')
        for i in range(epoch):
            sess.run(iterator.initializer)

            while True:
                try:

                    # t1 = time.time()
                    image_train, label_train = sess.run(next_element)
                    # t2 = time.time()
                    # print(counter, t2 - t1, "   next_element")
                    # print(image_train.shape, label_train.shape)
                    # print(label_train)
                    feed_dict = {images: image_train, labels: label_train}

                    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    # run_metadata = tf.RunMetadata()

                    # _, loss_val, acc_val, _ = sess.run([train_op, inference_loss, acc, inc_op], feed_dict=feed_dict,
                    #                                    options=options, run_metadata=run_metadata)

                    _, loss_val, acc_val, _ = sess.run([train_op, inference_loss, acc, inc_op], feed_dict=feed_dict)

                    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    # with open('timeline.json', 'w') as f:
                    #     f.write(chrome_trace)

                    # t3 = time.time()
                    # print(counter, t3 - t2, "   train_op, inference_loss, acc, inc_op")

                    counter += 1

                    if counter % 10 == 0:
                        time_h = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        log_c = 'time：' + time_h + ' counter:' + str(counter) + ' loss_val:' + str(
                            loss_val) + ' acc:' + str(acc_val)
                        print(type(log_c), log_c)
                        fp = open("iter_log.log", "a")
                        fp.write(log_c + "\n")
                        fp.close()

                    if counter % 10000 == 0:
                        filename = 'Face_vox_iter_{:d}'.format(counter) + '.ckpt'
                        filename = os.path.join(ckpt_save_dir, filename)
                        saver.save(sess, filename)

                except tf.errors.OutOfRangeError:
                    print('End of epoch %d', i)
                    break


if __name__ == '__main__':
    # $ python3 train.py --tfrecords '/Users/finup/Desktop/faces_emore/tfrecords/tran.tfrecords' --batch_size 64 --num_classes 85742 --lr [0.001, 0.0005, 0.0003, 0.0001] --ckpt_save_dir '/Users/finup/Desktop/faces_emore/face_real403_ckpt' --epoch 10000

    # tfrecords1 = '/Users/finup/Desktop/faces_emore/tfrecords/tran.tfrecords'
    # batch_size1 = 64
    # num_classes1 = 85742
    # lr1 = [0.001, 0.0005, 0.0003, 0.0001]
    # ckpt_save_dir1 = '/Users/finup/Desktop/faces_emore/face_real403_ckpt'
    # epoch1 = 10000
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # train(tfrecords1, batch_size1, lr1, ckpt_save_dir1, epoch1, num_classes1)

    parser = argparse.ArgumentParser()
    # Train
    parser.add_argument('--tfrecords', default='/Users/finup/Desktop/faces_emore/tfrecords/tran.tfrecords',
                        required=True)
    parser.add_argument('--batch_size', default=64, type=int, required=True)
    parser.add_argument('--num_classes', default=85742, type=int, required=True)
    parser.add_argument('--lr', default=[0.001, 0.0005, 0.0003, 0.0001], required=False)
    parser.add_argument('--ckpt_save_dir', default='/data/ChuyuanXiong/backup/face_real403_ckpt', required=True)
    parser.add_argument('--epoch', default=10000, type=int, required=False)
    parser.set_defaults(func=train)
    opt = parser.parse_args()
    opt.func(opt.tfrecords, opt.batch_size, opt.lr, opt.ckpt_save_dir, opt.epoch, opt.num_classes)
