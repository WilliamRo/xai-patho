from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class POLConfig(SmartTrainerHub):

  if_micro = Flag.boolean(False, 'if_micro', is_key=None)
  microimg_config = Flag.string(None, 'microimg_config', is_key=None)
  if_stain_norm = Flag.boolean(False, 'if_stain_norm', is_key=None)
  stain_method = Flag.string(None, 'stain_method', is_key=None)



# New hub class inherited from SmartTrainerHub must be registered
POLConfig.register()