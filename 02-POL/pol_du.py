from pol.pol_agent import POLSet, POLAgent
from roma import console
from tframe.data.augment.img_aug import image_augmentation_processor


def load_data():
  from pol_core import th

  datasets = POLAgent.load()
  if th.augmentation:
      datasets[0].batch_preprocessor = datasets[0].image_augmentation




  console.show_info('Data details')
  for ds in datasets:
    assert isinstance(ds, POLSet)

    # ds.fetch_data(ds)
    console.supplement(f'{ds.name}: {ds.features.shape})', level=2)

  return datasets
