from HyperNav.Utilities.Data.UVBase import Base, UVTimeList
from GeneralUtilities.Compute.list import TimeList, LatList, LonList, DepthList, flat_list
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
file_handler = FilePathHandler(ROOT_DIR,'GlobCurrent')

class GlobCurrentBase(Base):
	dataset_description = 'GlobCurrentHistorical'
	hours_list = np.arange(0,24,1).tolist()
	time_step = datetime.timedelta(hours=1)
	file_handler = file_handler
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)