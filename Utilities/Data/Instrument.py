from HyperNav.Utilities.Data.API import SiteAPI
from GeneralUtilities.Compute.list import TimeList, LatList, LonList
import datetime

class FloatBase():
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.time = TimeList([datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in self.time])
		assert isinstance(self.time,TimeList) 
		assert isinstance(self.lats,LatList) 
		assert isinstance(self.lons,LonList) 


	def return_float_pos_dict(self,date):
		date_idx = self.time.find_nearest(date,idx=True)
		lat = self.lats[date_idx]
		lon = self.lons[date_idx]
		time = self.time.seconds_since()[date_idx]
		return {'lat':lat,'lon':lon,'time':time}

class HI21A55(FloatBase):
	label='A55'
	time = ['2021-06-09 03:53:03','2021-06-09 19:01:19','2021-06-10 10:57:00','2021-06-11 07:39:10','2021-06-11 22:16:28',
	'2021-06-12 22:01:20','2021-06-13 21:52:14','2021-06-14 21:51:38','2021-06-15 21:47:54','2021-06-16 21:49:48','2021-06-17 21:47:43']
	lons = LonList([-156.4662,-156.4064,-156.3477,-156.2918,-156.2858,-156.2690,-156.3034,-156.3441,-156.4059,-156.4920,-156.5509])
	lats = LatList([19.5965, 19.5790, 19.5865, 19.6818, 19.7070, 19.7179, 19.7561, 19.7739, 19.7886, 19.7641, 19.6894])
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class HI21A54(FloatBase):
	label='A54'
	time = ['2021-06-09 02:56:43','2021-06-09 17:36:19','2021-06-10 08:21:58','2021-06-10 23:49:05','2021-06-11 22:16:13','2021-06-12 21:55:34',
		'2021-06-13 21:52:06','2021-06-14 21:50:34','2021-06-15 21:55:20','2021-06-16 21:49:24']
	lons = LonList([-156.5018,-156.4330,-156.3792,-156.3354,-156.2685,-156.1731,-156.1575,-156.1579,-156.1597,-156.1958])
	lats = LatList([19.5942, 19.5698, 19.5605, 19.5571, 19.5399, 19.5416, 19.5577, 19.6003, 19.6558, 19.6924])
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class HI21B55(FloatBase):
	label='B55'
	time = ['2021-06-18 01:19:00','2021-06-18 06:05:49','2021-06-18 21:45:54','2021-06-19 21:49:13','2021-06-20 21:49:30',
		'2021-06-21 21:48:56','2021-06-22 19:50:11']
	lons = LonList([-156.3901,-156.3991,-156.3896,-156.3452,-156.2445,-156.2115,-156.1658])
	lats = LatList([19.4567, 19.4448, 19.3951, 19.3190, 19.2701, 19.2193, 19.1824])
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class HI21B54(FloatBase):
	label='B54'
	time = ['2021-06-18 02:39:40','2021-06-18 07:58:47','2021-06-18 22:02:34','2021-06-19 21:49:02','2021-06-20 21:50:44',
		'2021-06-21 21:46:52','2021-06-22 19:56:07']
	lons = LonList([-156.3957,-156.3975,-156.3920,-156.2536,-156.1781,-156.1714,-156.1906])
	lats = LatList([19.4518,19.4327,19.3894,19.2756,19.3127,19.2821,19.2079])
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
