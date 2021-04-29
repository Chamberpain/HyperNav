from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import pandas as pd
import datetime
import geopy

gps_42_file = ROOT_DIR + '/../../Data/Navis-42/0042.gps.csv'
mission_42_file = ROOT_DIR + '/../../Data/Navis-42/0042.mission.csv'


class NavisParse(object):
	def __init__(self,gps_file,mission_file,start_idx = 0):
		self.index = start_idx
		df = pd.read_csv(gps_file)
		df = df.dropna(subset=['GPS Long','GPS Lat'])
		df = df[['Profile','Date','Time','GPS Lat','GPS Long']]
		time = [datetime.datetime.strptime(row[1].Date+' '+row[1].Time,'%b %d %Y %H:%M:%S') for row in df.iterrows()]
		pos = [geopy.Point(row[1]['GPS Lat'],row[1]['GPS Long']) for row in df.iterrows()]
		self.time_dict = dict(zip(df['Profile'].tolist(),time))
		self.pos_dict = dict(zip(df['Profile'].tolist(),pos))
		df = pd.read_csv(mission_file)
		self.park_pressure_dict = dict(zip(df['Profile'].tolist(),df['ParkPressure'].tolist()))

	def make_float_pos_dict(self):
		float_pos_dict = {'datetime':self.time_dict[self.index],
		'lat':self.pos_dict[self.index].latitude,
		'lon':self.pos_dict[self.index].longitude,
		'profile':self.index}

	def increment_profile(self):
		self.index+=1
		return (self.make_float_pos_dict(),self.park_pressure_dict[self.index])