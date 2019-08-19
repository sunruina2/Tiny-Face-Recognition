# Tiny-Face-Recognition

Extract an complete process from insightface_tf to do face recognition and verification using pure TensorFlow.

### Dataset

Training and Testing Dataset Download Website: [Baidu](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ)

Contains:

* train.rec/train.idx   : Main training data
* \*.bin  : varification data

### Examples

Make TFRecords File:

```
$ python3 mx2tfrecords.py --bin_path '/Users/finup/Desktop/faces_emore/train.rec' --idx_path '/Users/finup/Desktop/faces_emore/train.idx' --tfrecords_file_path '/Users/finup/Desktop/faces_emore/tfrecords'
```


Train:

```
$ python3 train.py --tfrecords '/Users/finup/Desktop/faces_emore/tfrecords/tran.tfrecords' --batch_size 64 --num_classes 85742 --lr [0.001, 0.0005, 0.0003, 0.0001] --ckpt_save_dir '/Users/finup/Desktop/faces_emore//face_real403_ckpt' --epoch 10000
```

Test:

```
$ python3 eval_veri.py --datasets '/Users/finup/Desktop/faces_emore/cfp_fp.bin' --dataset_name 'cfp_fp' --num_classes 85742 --ckpt_restore_dir '/Users/finup/Desktop/faces_emore/Face_vox_iter_78900.ckpt'
```


# Results

Datasets|backbone| loss|steps|batch_size|acc
-------|--------|-----|---|-----------|----|
lfw    | resnet50 | ArcFace | 78900 | 64 | 0.9903
cfp_ff | resnet50 | ArcFace | 78900 | 64 | 0.9847
cfp_fp | resnet50 | ArcFace | 78900 | 64 | 0.8797
agedb_30| resnet50 | ArcFace | 78900|64 | 0.8991

Limited by the training time, so I just release the half-epoch training results temporarily. The model will be optimized later.






# References

1. [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
2. [InsightFace_MxNet](https://github.com/deepinsight/insightface)

```
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```

```
<tf.Tensor 'gradients/conv1_1/3x3_s1/Conv2D_grad/tuple/control_dependency_1:0' shape=(7, 7, 3, 64) dtype=float32>, <tf.Variable 'conv1_1/3x3_s1/kernel:0' shape=(7, 7, 3, 64) dtype=float32_ref>

<tf.Tensor 'gradients/bn1_1/3x3_s1/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'bn1_1/3x3_s1/gamma:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/bn1_1/3x3_s1/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(64,) dtype=float32>, <tf.Variable 'bn1_1/3x3_s1/beta:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1a_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 64, 64) dtype=float32>, <tf.Variable 'conv2_1a_1x1_reduce/kernel:0' shape=(1, 1, 64, 64) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1a_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1a_1x1_reduce/gamma:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1a_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1a_1x1_reduce/beta:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1a_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 64, 64) dtype=float32>, <tf.Variable 'conv2_1a_3x3/kernel:0' shape=(3, 3, 64, 64) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1a_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1a_3x3/gamma:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1a_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1a_3x3/beta:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1a_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 64, 256) dtype=float32>, <tf.Variable 'conv2_1a_1x1_increase/kernel:0' shape=(1, 1, 64, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1a_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn2_1a_1x1_increase/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1a_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn2_1a_1x1_increase/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1a_1x1_shortcut/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 64, 256) dtype=float32>, <tf.Variable 'conv2_1a_1x1_shortcut/kernel:0' shape=(1, 1, 64, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1a_1x1_shortcut/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn2_1a_1x1_shortcut/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1a_1x1_shortcut/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn2_1a_1x1_shortcut/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1b_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 256, 64) dtype=float32>, <tf.Variable 'conv2_1b_1x1_reduce/kernel:0' shape=(1, 1, 256, 64) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1b_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1b_1x1_reduce/gamma:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1b_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1b_1x1_reduce/beta:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1b_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 64, 64) dtype=float32>, <tf.Variable 'conv2_1b_3x3/kernel:0' shape=(3, 3, 64, 64) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1b_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1b_3x3/gamma:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1b_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1b_3x3/beta:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1b_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 64, 256) dtype=float32>, <tf.Variable 'conv2_1b_1x1_increase/kernel:0' shape=(1, 1, 64, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1b_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn2_1b_1x1_increase/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1b_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn2_1b_1x1_increase/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1c_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 256, 64) dtype=float32>, <tf.Variable 'conv2_1c_1x1_reduce/kernel:0' shape=(1, 1, 256, 64) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1c_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1c_1x1_reduce/gamma:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1c_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1c_1x1_reduce/beta:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1c_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 64, 64) dtype=float32>, <tf.Variable 'conv2_1c_3x3/kernel:0' shape=(3, 3, 64, 64) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1c_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1c_3x3/gamma:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1c_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(64,) dtype=float32>, <tf.Variable 'bn2_1c_3x3/beta:0' shape=(64,) dtype=float32_ref>

<tf.Tensor 'gradients/conv2_1c_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 64, 256) dtype=float32>, <tf.Variable 'conv2_1c_1x1_increase/kernel:0' shape=(1, 1, 64, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1c_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn2_1c_1x1_increase/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn2_1c_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn2_1c_1x1_increase/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2a_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 256, 128) dtype=float32>, <tf.Variable 'conv3_2a_1x1_reduce/kernel:0' shape=(1, 1, 256, 128) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2a_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2a_1x1_reduce/gamma:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2a_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2a_1x1_reduce/beta:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2a_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 128, 128) dtype=float32>, <tf.Variable 'conv3_2a_3x3/kernel:0' shape=(3, 3, 128, 128) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2a_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2a_3x3/gamma:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2a_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2a_3x3/beta:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2a_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 128, 512) dtype=float32>, <tf.Variable 'conv3_2a_1x1_increase/kernel:0' shape=(1, 1, 128, 512) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2a_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2a_1x1_increase/gamma:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2a_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2a_1x1_increase/beta:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2a_1x1_shortcut/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 256, 512) dtype=float32>, <tf.Variable 'conv3_2a_1x1_shortcut/kernel:0' shape=(1, 1, 256, 512) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2a_1x1_shortcut/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2a_1x1_shortcut/gamma:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2a_1x1_shortcut/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2a_1x1_shortcut/beta:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2b_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 512, 128) dtype=float32>, <tf.Variable 'conv3_2b_1x1_reduce/kernel:0' shape=(1, 1, 512, 128) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2b_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2b_1x1_reduce/gamma:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2b_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2b_1x1_reduce/beta:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2b_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 128, 128) dtype=float32>, <tf.Variable 'conv3_2b_3x3/kernel:0' shape=(3, 3, 128, 128) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2b_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2b_3x3/gamma:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2b_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2b_3x3/beta:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2b_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 128, 512) dtype=float32>, <tf.Variable 'conv3_2b_1x1_increase/kernel:0' shape=(1, 1, 128, 512) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2b_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2b_1x1_increase/gamma:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2b_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2b_1x1_increase/beta:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2c_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 512, 128) dtype=float32>, <tf.Variable 'conv3_2c_1x1_reduce/kernel:0' shape=(1, 1, 512, 128) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2c_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2c_1x1_reduce/gamma:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2c_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2c_1x1_reduce/beta:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2c_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 128, 128) dtype=float32>, <tf.Variable 'conv3_2c_3x3/kernel:0' shape=(3, 3, 128, 128) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2c_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2c_3x3/gamma:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2c_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2c_3x3/beta:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2c_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 128, 512) dtype=float32>, <tf.Variable 'conv3_2c_1x1_increase/kernel:0' shape=(1, 1, 128, 512) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2c_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2c_1x1_increase/gamma:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2c_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2c_1x1_increase/beta:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2d_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 512, 128) dtype=float32>, <tf.Variable 'conv3_2d_1x1_reduce/kernel:0' shape=(1, 1, 512, 128) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2d_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2d_1x1_reduce/gamma:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2d_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2d_1x1_reduce/beta:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2d_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 128, 128) dtype=float32>, <tf.Variable 'conv3_2d_3x3/kernel:0' shape=(3, 3, 128, 128) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2d_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2d_3x3/gamma:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2d_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(128,) dtype=float32>, <tf.Variable 'bn3_2d_3x3/beta:0' shape=(128,) dtype=float32_ref>

<tf.Tensor 'gradients/conv3_2d_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 128, 512) dtype=float32>, <tf.Variable 'conv3_2d_1x1_increase/kernel:0' shape=(1, 1, 128, 512) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2d_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2d_1x1_increase/gamma:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/bn3_2d_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(512,) dtype=float32>, <tf.Variable 'bn3_2d_1x1_increase/beta:0' shape=(512,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3a_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 512, 256) dtype=float32>, <tf.Variable 'conv4_3a_1x1_reduce/kernel:0' shape=(1, 1, 512, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3a_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3a_1x1_reduce/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3a_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3a_1x1_reduce/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3a_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 256, 256) dtype=float32>, <tf.Variable 'conv4_3a_3x3/kernel:0' shape=(3, 3, 256, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3a_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3a_3x3/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3a_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3a_3x3/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3a_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 256, 1024) dtype=float32>, <tf.Variable 'conv4_3a_1x1_increase/kernel:0' shape=(1, 1, 256, 1024) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3a_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(1024,) dtype=float32>, <tf.Variable 'bn4_3a_1x1_increase/gamma:0' shape=(1024,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3a_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(1024,) dtype=float32>, <tf.Variable 'bn4_3a_1x1_increase/beta:0' shape=(1024,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3a_1x1_shortcut/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 512, 1024) dtype=float32>, <tf.Variable 'conv4_3a_1x1_shortcut/kernel:0' shape=(1, 1, 512, 1024) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3a_1x1_shortcut/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(1024,) dtype=float32>, <tf.Variable 'bn4_3a_1x1_shortcut/gamma:0' shape=(1024,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3a_1x1_shortcut/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(1024,) dtype=float32>, <tf.Variable 'bn4_3a_1x1_shortcut/beta:0' shape=(1024,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3b_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 1024, 256) dtype=float32>, <tf.Variable 'conv4_3b_1x1_reduce/kernel:0' shape=(1, 1, 1024, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3b_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3b_1x1_reduce/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3b_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3b_1x1_reduce/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3b_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 256, 256) dtype=float32>, <tf.Variable 'conv4_3b_3x3/kernel:0' shape=(3, 3, 256, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3b_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3b_3x3/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3b_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3b_3x3/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3b_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 256, 1024) dtype=float32>, <tf.Variable 'conv4_3b_1x1_increase/kernel:0' shape=(1, 1, 256, 1024) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3b_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(1024,) dtype=float32>, <tf.Variable 'bn4_3b_1x1_increase/gamma:0' shape=(1024,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3b_1x1_increase/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(1024,) dtype=float32>, <tf.Variable 'bn4_3b_1x1_increase/beta:0' shape=(1024,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3c_1x1_reduce/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 1024, 256) dtype=float32>, <tf.Variable 'conv4_3c_1x1_reduce/kernel:0' shape=(1, 1, 1024, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3c_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3c_1x1_reduce/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3c_1x1_reduce/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3c_1x1_reduce/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3c_3x3/Conv2D_grad/tuple/control_dependency_1:0' shape=(3, 3, 256, 256) dtype=float32>, <tf.Variable 'conv4_3c_3x3/kernel:0' shape=(3, 3, 256, 256) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3c_3x3/FusedBatchNorm_grad/tuple/control_dependency_1:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3c_3x3/gamma:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/bn4_3c_3x3/FusedBatchNorm_grad/tuple/control_dependency_2:0' shape=(256,) dtype=float32>, <tf.Variable 'bn4_3c_3x3/beta:0' shape=(256,) dtype=float32_ref>

<tf.Tensor 'gradients/conv4_3c_1x1_increase/Conv2D_grad/tuple/control_dependency_1:0' shape=(1, 1, 256, 1024) dtype=float32>, <tf.Variable 'conv4_3c_1x1_increase/kernel:0' shape=(1, 1, 256, 1024) dtype=float32_ref>)

```

