import datetime
import urllib.request
import ast

def return_float_pos_dict():
	address = 'http://204.197.4.164/HyperNAV/api/v1/0042/meta'
	with urllib.request.urlopen(address) as response:
	   html = response.read()
	pos_dict = ast.literal_eval(html.decode('utf-8'))
	pos_dict['datetime'] = datetime.datetime.strptime(pos_dict['datetime'], '%Y-%m-%dT%H:%M:%S+00:00')
	return pos_dict