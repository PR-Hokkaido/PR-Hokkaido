#  -*- coding: utf-8 -*-
#  migrationの次に使う画像拡張用クラス

import os
import numpy as np
import skimage.io
import skimage.transform
from skimage.transform import AffineTransform, warp
from skimage.transform import resize, SimilarityTransform
import warnings


class Augment:
    warnings.filterwarnings("ignore", category=UserWarning)  # windowsだと非常に多く警告が出てくるので、表示させないようにする

    #  画像読み込み
    def load(paths_train, width, height):
        images = []
        imagenames = []
        labels = []

        for i, path in enumerate(paths_train):
            # resize:image=ndarray output_shape=tupleもしくはndarray return:ndarray
            image = resize(skimage.io.imread(path), (width, height), mode="constant")
            imagename = os.path.basename(path)
            label = os.path.basename(os.path.dirname(path))
            images.append(image)
            imagenames.append(imagename)
            labels.append(label)

        return images, imagenames, labels

    #  変換実行
    def fast_warp(img, tf, output_shape=(50, 50), mode='constant', order=1):
        m = tf.params
        return warp(img, m, output_shape=output_shape, mode=mode, order=order)

    def build_centering_transform(image_shape, target_shape=(50, 50)):
        if len(image_shape) == 2:
            rows, cols = image_shape
        else:
            rows, cols, _ = image_shape
        trows, tcols = target_shape
        shift_x = (cols - tcols) / 2.0
        shift_y = (rows - trows) / 2.0
        return SimilarityTransform(translation=(shift_x, shift_y))

    def build_center_uncenter_transforms(image_shape):
        center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5
        tform_uncenter = SimilarityTransform(translation=-center_shift)
        tform_center = SimilarityTransform(translation=center_shift)
        return tform_center, tform_uncenter

    def build_transform(zoom=(1.0, 1.0), rot=0, shear=0, trans=(0, 0), flip=False):
        if flip:
            shear += 180
            rot += 180

        r_rad = np.deg2rad(rot)
        s_rad = np.deg2rad(shear)
        tform_augment = AffineTransform(scale=(1/zoom[0], 1/zoom[1]), 
                                        rotation=r_rad, shear=s_rad, translation=trans)
        return tform_augment

    # 一定の範囲内（引数の範囲内）で、ランダムに変形した画像を生成する
    def random_transform(self, zoom_range, rotation_range, shear_range, 
                         translation_range, do_flip=True, allow_stretch=False, rng=np.random):
        shift_x = rng.uniform(*translation_range)
        shift_y = rng.uniform(*translation_range)
        translation = (shift_x, shift_y)

        rotation = rng.uniform(*rotation_range)
        shear = rng.uniform(*shear_range)

        if do_flip:  # フリップ処理（引数で指定されているなら、50%の確率で行う
            flip = (rng.randint(2) > 0)
        else:
            flip = False

        log_zoom_range = [np.log(z) for z in zoom_range]

        if isinstance(allow_stretch, float):
            log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
            zoom = np.exp(rng.uniform(*log_zoom_range))
            stretch = np.exp(rng.uniform(*log_stretch_range))
            z_x = zoom * stretch
            z_y = zoom / stretch
        elif allow_stretch is True:
            z_x = np.exp(rng.uniform(*log_zoom_range))
            z_y = np.exp(rng.uniform(*log_zoom_range))
        else:
            z_x = z_y = np.exp(rng.uniform(*log_zoom_range))

        return self.build_transform((z_x, z_y), rotation, shear, translation, flip)

    #  変換のパラメータを作成 t_shape:width heightのtuple
    def perturb(self, img, augmentation_params, t_shape=(50, 50), rng=np.random):
        tf_centering = self.build_centering_transform(img.shape, t_shape)  # line44 img.shapeで横幅と縦幅が返る
        tf_center, tf_uncenter = self.build_center_uncenter_transforms(img.shape)
        tf_aug = self.random_transform(self, rng=rng, **augmentation_params)
        tf_aug = tf_uncenter + tf_aug + tf_center
        tf_aug = tf_centering + tf_aug
        warp_one = self.fast_warp(img, tf_aug, output_shape=t_shape, mode='constant')
        return warp_one.astype('float32')
