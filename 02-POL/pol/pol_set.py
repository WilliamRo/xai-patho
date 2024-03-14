from pictor import Pictor
from tframe import DataSet
import cv2
import numpy as np

class POLSet(DataSet):
  def __init__(self, subjects=None, **kwargs):
    # Call parent's constructor
    super().__init__(**kwargs)

    # Add new attribute
    self.subjects = subjects

  def show(self):
    da = Pictor.image_viewer(title=f'{self.name}')
    da.objects = self.features
    # da.labels = [self.properties[pedia.classes][c]
    #                     for c in self.dense_labels]
    da.show()

  def batch_preprocessor_map(self, data_batch, is_training):
    """ Uniform image size """
    maxh = max([f[0].shape[0] for f in data_batch.features])
    maxw = max([f[0].shape[1] for f in data_batch.features])
    features = []
    for img in data_batch.features:
      img = img[0]
      w, h = img.shape[1], img.shape[0]
      threshold, threshold_ratio = 200, 0.2
      random_x = np.random.randint(0, maxw - w) if maxw - w else 0
      random_y = np.random.randint(0, maxh - h) if maxh - h else 0

      # padding
      bg = np.full((maxh, maxw, 3), 0, dtype=np.float_)
      bg[random_y:random_y + h, random_x:random_x + w] = img
      features.append(bg)
      cv2.imwrite(rf'G:\figure\{random_x}.png', bg)
    data_batch.features = np.array(features)
    return data_batch

  def get_selected_batch(self, batch_size):
    num = [0, 0]
    features, targets = [], []
    while num[0]+num[1] < batch_size:
      index = np.random.randint(self.features.shape[0])
      if (self.targets[index] == [1, 0]).all() and num[0]<batch_size//2:
        features.append(self.features[index])
        targets.append(self.targets[index])
        num[0] += 1
      elif (self.targets[index] == [0, 1]).all() and num[1]<batch_size//2:
        features.append(self.features[index])
        targets.append(self.targets[index])
        num[1] += 1
    features = np.array(features)
    targets = np.array(targets)
    return DataSet(features, targets)

  def batch_preprocessor1(self, data_batch, is_training):
    from pol_core import th

    _, _, win = th.data_config.split(':')
    win = int(win)
    features, targets, note = (data_batch.features, data_batch.targets,
                               data_batch.data_dict['note'])
    patches, lables, indices = [], [], []
    for i, img in enumerate(features):
      new_patches, num = self.get_noted_patch(img[0], win, note[i][0])
      # print(img[0].shape, new_patches.shape,i)
      # patches.append(np.array(new_patches))
      # lables.append([targets[i]] * num)
      # indices.append([i] * num)

      if len(new_patches.shape) != 4:
        a = 0
      if i == 0:
        patches, lables, indices = new_patches, [targets[i]] * num, [i] * num
      else:
        patches = np.concatenate((patches, new_patches))
        lables = np.concatenate((lables, [targets[i]] * num))
        indices = np.concatenate((indices, [i] * num))

    data_batch.data_dict['features'] = np.array(patches)
    # data_batch.data_dict['targets'] = np.array(lables)
    data_batch.data_dict['indices'] = np.array(indices)
    # print(patches.shape)
    del data_batch.data_dict['note']

    # show_batch = WSISet(np.array(patches), np.array(lables))
    # show_batch.show()
    return data_batch

  def get_patch(self, img, win):
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    patches = []
    w, h = img.shape[1], img.shape[0]
    w_num, h_num = (w + win - 1) // win, (h + win - 1) // win
    threshold, threshold_ratio = 200, 0.2
    random_x = np.random.randint(0, w_num * win - w) if w_num * win - w else 0
    random_y = np.random.randint(0, h_num * win - h) if h_num * win - h else 0

    # padding
    bg = np.full((win * h_num, win * w_num, 3), 255, dtype=np.uint8)
    bg[random_y:random_y + h, random_x:random_x + w] = img

    # partitioning into blocks
    for x in range(w_num):
      for y in range(h_num):
        new = bg[y * win:y * win + win, x * win:x * win + win]

        # select
        patch = new
        gray_image = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold, 255,
                                        cv2.THRESH_BINARY_INV)
        colored_ratio = cv2.countNonZero(binary_image) / win / win
        if colored_ratio < threshold_ratio and (w_num * h_num != 1):
          # cv2.rectangle(bg, (x * win, y * win), (x * win + win, y * win + win),
          #               color=(128, 128, 128), thickness=-1)
          continue
        patches.append(new)
        # cv2.rectangle(bg, (x * win, y * win), (x * win + win, y * win + win),
        #               color=(0, 0, 0), thickness=3)
    # cv2.imshow("Image", bg)
    # cv2.waitKey(0)
    # patches = img
    return np.array(patches), len(patches)

  def get_noted_patch(self, img, win, all_areas):
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    patches = []
    w0, h0, w1, h1 = 0, 0, img.shape[1], img.shape[0]
    threshold, threshold_ratio = 200, 0.2

    # partitioning into blocks
    for x in range(w0, w1, win):
      for y in range(w0, h1, win):
        patch = img[y:y + win, x:x + win]
        if x+win > w1 or y+win > h1:
          blank_area = np.full((win, win, 3), 255, dtype=np.uint8)
          blank_area[:patch.shape[0], :patch.shape[1]] = patch
          patch = blank_area

        # select
        if self.if_noted(x, y, win, all_areas):
          patches.append(patch)
          # cv2.rectangle(img, (x, y ), (x + win, y + win),
          #               color=(0, 0, 0), thickness=3)

    # plot note
    # for i, anno_name in enumerate(all_areas):
    #   for points in all_areas[anno_name]:
    #     points.append(points[0])
    #     points_np = np.array(points, dtype=np.int32)
    #     cv2.polylines(img, [points_np], isClosed=True, color=(0, 0, 0), thickness=10)

    # # 显示
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    return np.array(patches), len(patches)


  def image_augmentation(self, data_batch: DataSet, is_training: bool):
    import imgaug as ia
    from imgaug import augmenters as iaa
    import random
    random_int = random.randint(0, 10000)
    """
    random rotate/flip/GaussianMoise/GaussianBlur/crop/
           Brightness/Hue_Saturation/Temperature
    """

    images = data_batch.features
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)

    seq = iaa.Sequential([
      iaa.Fliplr(0.5),  # horizontal flips
      iaa.Flipud(0.5),
      iaa.Crop(px=(0, 16)),
      sometimes(iaa.GaussianBlur(sigma=(0, 3.0)))
      # sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
      # sometimes(iaa.Affine(
      #   scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
      #   translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
      #   rotate=(-180, 180),
      #   shear=(-8, 8)
      # )),
      # iaa.GaussianBlur(sigma=(0, 5.0)),
      # iaa.OneOf([
      #   iaa.LinearContrast((0.75, 1.5)),
      #   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
      #                             per_channel=0.5),
      #   iaa.Multiply((0.5, 1.5), per_channel=0.5),
      # ]),
    ], random_order=True)  # apply augmenters in random order

    data_batch.features = seq(images=images)
    # ia.imshow(ia.draw_grid(list(images_aug), cols=4, rows=8))
    # rows, cols = 4, 4
    # fig, axs = plt.subplots(rows, cols, figsize=(16, 8))
    # for i in range(rows):
    #   for j in range(cols):
    #     axs[i, j].imshow(data_batch.features[i * cols + j])
    #     axs[i, j].axis('off')
    #
    # # 保存图像
    # plt.savefig(rf'C:\Data\polyp\train_imgaug\{random_int}.png', dpi=300)

    return data_batch
