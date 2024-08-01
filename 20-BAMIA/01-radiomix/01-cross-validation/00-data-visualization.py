from pictor.xomics.omix import Omix



data_path = r'../data/radiomics-111x851.omix'
omix = Omix.load(data_path)
omix.show_in_explorer()

