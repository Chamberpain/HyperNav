from GeneralUtilities.Filepath.instance import FilePathHandler
from HyperNav.Data.__init__ import ROOT_DIR

class HypernavFileHandler(FilePathHandler):
    """"" Extension of the basic file path handler class with specific filepaths relevent to the Hypernav project
    """""
    @staticmethod
    def nc_file(filename = 'Uniform_out'):
        return ROOT_DIR + '/../Pipeline/Compute/RunParcels/tmp/'+filename+'.nc'
