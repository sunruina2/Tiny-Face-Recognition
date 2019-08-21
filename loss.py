import tensorflow as tf
import math


def arcface_loss(embedding, labels, out_num, w_init=None, s=64.,
                 m=0.5):  # embedding=emb, labels=labels, w_init=w_init_method, out_num=num_classes
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)  # cos(θ+m)=cosθcosm−sinθsinm
    # mm = sin_m * m  # issue 1 ?
    threshold = math.cos(math.pi - m)  # ?
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1,
                                 keep_dims=True)  # 特征值归一化，(？,1),默认情况下是计算欧氏距离的L2范数(向量的摩长||x||，向量点到原点的距离), axis为按哪个维度计算范数，可取0或1，0代表列，1代表行
        embedding = tf.div(embedding, embedding_norm,
                           name='norm_embedding')  # (？,512)/(？,1) = (？,512)，emb/||x|| = emb_new, ||emb_new||=1,只保留方向向量，摩长=1
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init,
                                  dtype=tf.float32)  # 如果已存在参数定义相同的变量，就返回已存在的变量，否则创建由参数定义的新变量。w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)这个初始化器是用来保持每一层的梯度大小都差不多相同。
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)  # (1, 85742) ,特征值取出后进行归一化，去除长度影响，只保留角度信息。
        weights = tf.div(weights, weights_norm, name='norm_weights')  # (512, 85742)/(1, 85742)=(512, 85742)
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')  # 点乘 x*w =(？,512) *(512, 85742) = (?,85742),由于x和权重都做了归一化所以，X*W=|x|*|w|cos = |1||1|cos=cos
        cos_t2 = tf.square(cos_t, name='cos_2')  # (?,85742)
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')  # (?,85742) (1)三角函数公式 sin方=1-cos方
        sin_t = tf.sqrt(sin_t2, name='sin_t')  # (?,85742) ，(2)从而得到sin = 根号(sin2)
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m),
                                 name='cos_mt')  # (?,85742) cos(t+m)=cos_t * cos_m - sin_t * sin_m ，即coscos - sinsin

        # this condition controls the theta+m should in range [0, pi] ？？？
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold  # (?,85742)
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)  # 布尔量，(?,85742)，正数为True，其他为False

        keep_val = s * (cos_t - sin_m * m)  # (?,85742)  ？？
        cos_mt_temp = tf.where(cond, cos_mt,
                               keep_val)  # 在cos_mt里面挑选的cond为True位置，在keep_val里面挑选的cond为Falese位置，对应位置放数-组成新的矩阵

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')  # (?,85742)
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')  # (?,85742)

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')  # (?,85742)扩大球面面积，是的各类别label分布更分散，从而更好找到边界

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask),
                        name='arcface_loss_output')  # (?,85742)
    return output
