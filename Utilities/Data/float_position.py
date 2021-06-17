import datetime
import urllib.request
import ast

def return_float_pos_dict(address):
	with urllib.request.urlopen(address) as response:
	   html = response.read()
	pos_dict = ast.literal_eval(html.decode('utf-8'))
	pos_dict['datetime'] = datetime.datetime.strptime(pos_dict['datetime'], '%Y-%m-%dT%H:%M:%S+00:00')
	pos_dict['ID']=address.split('/')[-2]
	return pos_dict

def return_float_pos_list():
	out = []
	out.append(return_float_pos_dict('http://204.197.4.164/HyperNAV/api/v1/0054/meta'))
	out.append(return_float_pos_dict('http://204.197.4.164/HyperNAV/api/v1/0055/meta'))
	return out

manual_position55 = [
					{'ID':'0055','profile':1,'lat':19.5965,'lon':-156.4662,'datetime':datetime.datetime(2021,6,9,3,53)},
					{'ID':'0055','profile':2,'lat':19.5790,'lon':-156.4064,'datetime':datetime.datetime(2021,6,9,19,1)},
					{'ID':'0055','profile':3,'lat':19.5865,'lon':-156.3477,'datetime':datetime.datetime(2021,6,10,10,57)},
					{'ID':'0055','profile':4,'lat':19.6818,'lon':-156.2918,'datetime':datetime.datetime(2021,6,11,7,39)},
					{'ID':'0055','profile':5,'lat':19.7070,'lon':-156.2858,'datetime':datetime.datetime(2021,6,11,22,16)},
					{'ID':'0055','profile':6,'lat':19.7179,'lon':-156.2690,'datetime':datetime.datetime(2021,6,12,22,1)},
					]
