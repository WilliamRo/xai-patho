from collections import Counter
from pol_set import POLSet
from roma import finder
from sklearn.model_selection import train_test_split, StratifiedKFold
from tframe.data.base_classes import DataAgent
from tframe import console
from tframe.utils.misc import convert_to_one_hot, convert_to_dense_labels

import cv2
import glob
import numpy as np
import os
import pandas as pd
import pickle



class POLAgent(DataAgent):

  @classmethod
  def load(cls):
    from pol_core import th
    dataset_name, _, win = th.data_config.split(':')
    dataset_path = os.path.join(th.data_dir, '02-POL/patch_pair')

    file_name = (fr"{th.data_config.replace(':', '_')}_{th.input_shape[0]}"
                 fr"{'_' + th.stain_method if th.if_stain_norm else ''}_"
                 fr"{'_'+str(th.folds_k)+'_'+str(th.folds_i) if th.folds_i else ''}.ds")
    file_path = os.path.join(dataset_path, file_name)

    if not os.path.exists(file_path):
      # (0) setting patch path
      patch_path = os.path.join(dataset_path, win)

      # (1) reading cross-validation csv
      folds_file_path = os.path.join(dataset_path, rf'{win}_folds_{th.folds_k}')
      if not os.path.exists(folds_file_path):
        split_img(patch_path, th.folds_k, folds_file_path)
      csv_file_path = os.path.join(folds_file_path, f"split_results_{th.folds_i}.csv")
      df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

      # (2) split dataset and save
      train_ds, val_ds, test_ds = cls.load_as_tframe_data(patch_path, df)
      with open(file_path, 'wb') as file:
        console.show_status(f'saving {file_path}...')
        pickle.dump((train_ds, val_ds, test_ds), file)
    else:
      with open(file_path, 'rb') as _input_:
        console.show_status(f'loading {file_path}...')
        train_ds, val_ds, test_ds = pickle.load(_input_)

    # ds = train_ds.split(7, 1, names=['Train-Set', 'Val-Set'],
    #                     over_classes=True, random=True)
    # ds += (test_ds,)
    train_ds.name = 'Train-Set'
    val_ds.name = 'Val-Set'
    test_ds.name = 'Test-Set'
    ds = train_ds, val_ds, test_ds
    return ds



  @classmethod
  def load_as_tframe_data(cls, patch_path, df):
    from pol_core import th
    # down_samples 1 4 16

    cn = ['HP', 'TA']
    dataset_name, down_samples, win = th.data_config.split(':')

    ds = []
    for set in ['Train', 'Validation', 'Test']:
      features, targets= [], []
      set_df = df[df['Set'] == set]
      names = set_df['Name']
      for name in names:
        img_paths = finder.walk(
          patch_path, pattern=f'*{name}*{dataset_name}.png')

        for img_path in img_paths:
          type_name = os.path.basename(img_path).split('_')[0]
          img = cv2.imread(img_path)
          if th.input_shape[0] != win:
            img = cv2.resize(img, tuple(th.input_shape[:2]))
          features.append(img)
          targets.append(cn.index(type_name))

      targets = convert_to_one_hot(targets, len(cn))
      PROPERTIES = {
        'CLASSES': cn,
        'NUM_CLASSES': len(cn)
      }
      ds.append(
        POLSet(features=np.array(features), targets=targets, **PROPERTIES))

    return ds



def split_img(v_path, k, save_folder):
  '''
    train:val = 7:1
  '''
  if not os.path.exists(save_folder): os.makedirs(save_folder)
  names = [(0 if os.path.basename(img_p).split('_')[0] == 'HP' else 1,
            os.path.basename(img_p).split('_')[1])
           for img_p in glob.glob(os.path.join(v_path, '*.png'))]
  unique_list = []
  for sublist in names:
    if sublist not in unique_list:
      unique_list.append(sublist)

  df = pd.DataFrame(unique_list, columns=["Label", "Name"])

  skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

  for i, (train_index, test_index) in enumerate(
      skf.split(df["Name"], df["Label"]), 1):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]
    train_df, val_df = train_test_split(train_set, test_size=0.125,
                                         stratify=train_set["Label"],
                                         random_state=42)

    combined_set = pd.concat(
      [train_df.assign(Set="Train"), val_df.assign(Set="Validation"),
       test_set.assign(Set="Test")])

    # 保存结果 DataFrame 到 CSV 文件
    csv_file_path = os.path.join(save_folder, f"split_results_{i}.csv")
    combined_set.to_csv(csv_file_path, index=True)



if __name__ == '__main__':
    from pol_core import th
    th.data_config = 'wsi:16:300'


    th.num_classes = 2
    th.if_stain_norm = False
    th.stain_method = 'reinhard'

    th.folds_k = 10
    th.folds_i = 2
    th.input_shape = [100, 100, 3]


    dirr = ['C:\Data\polyp', r'D:\2d\xai-omics\data\tif']
    datasets = POLAgent.load()


