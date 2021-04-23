from netCDF4 import Dataset
import os

def token_adjust(dataset_token,variable):
	dataset_token[variable][:,:,0,:] = 0
	dataset_token[variable][:,:,1,:] = 0
	dataset_token[variable][:,:,-1,:] = 0
	dataset_token[variable][:,:,-2,:] = 0
	dataset_token[variable][:,:,:,0] = 0
	dataset_token[variable][:,:,:,1] = 0
	dataset_token[variable][:,:,:,-1] = 0
	dataset_token[variable][:,:,:,-2] = 0
	return dataset_token


base_folder = '/Users/pchamberlain/Data/Raw/PACIOOS/february/'
for file_ in os.listdir(base_folder):
	if file_.endswith('.nc'):
		token = Dataset(os.path.join(base_folder,file_),'r+')
		token = token_adjust(token,'u')
		token = token_adjust(token,'v')
		token.close()