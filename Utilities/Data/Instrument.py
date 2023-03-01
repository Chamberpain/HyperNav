from HyperNav.Utilities.Data.API import SiteAPI
from GeneralUtilities.Compute.list import TimeList, LatList, LonList,DepthList
import datetime
from GeneralUtilities.Data.Lagrangian.Argo.array_class import ArgoArray
import shapely.geometry
import geopy


class FloatBase():
	def __init__(self,*args,**kwargs):
		assert isinstance(self.time,TimeList) 
		assert isinstance(self.lats,LatList) 
		assert isinstance(self.lons,LonList) 

	def return_float_pos_dict(self,time):
		idx = self.time.index(time)
		start_time = self.time[idx].timestamp()
		end_time = self.time[idx+1].timestamp()
		lat = self.lats[idx]
		lon = self.lons[idx]
		drift_depth = self.drift_depths[idx]
		vertical_speed = 0.076
		total_cycle_time = (self.time[idx+1]-self.time[idx]).seconds
		argo_cfg = {'lat': lat, 'lon': lon, 'target_lat': self.lats[idx+1], 'target_lon': self.lons[idx+1],
			'time': start_time, 'end_time': end_time, 'depth': 10,'min_depth': 10, 'drift_depth': abs(drift_depth), 'max_depth':abs(self.max_depth),
			 'surface_time': self.surface_time,'total_cycle_time': total_cycle_time,'vertical_speed':vertical_speed,
			 }
		return argo_cfg
	def return_float_pos_dict_list(self,ocean_shape):
		dict_list = []
		for k in range(len(self.lons)-1):
			if ocean_shape.contains(shapely.geometry.Point(self.lons[k],self.lats[k]))&ocean_shape.contains(shapely.geometry.Point(self.lons[k+1],self.lats[k+1])):
				start_time = self.time[k].timestamp()
				end_time = self.time[k+1].timestamp()
				lat = self.lats[k]
				lon = self.lons[k]
				drift_depth = self.drift_depths[k]
				vertical_speed = 0.076
				total_cycle_time = (self.time[k+1]-self.time[k]).seconds
				argo_cfg = {'lat': lat, 'lon': lon, 'target_lat': self.lats[k+1], 'target_lon': self.lons[k+1],
					'time': start_time, 'end_time': end_time, 'depth': 10,'min_depth': 10, 'drift_depth': abs(drift_depth), 'max_depth':abs(self.max_depth),
					 'surface_time': self.surface_time,'total_cycle_time': total_cycle_time,'vertical_speed':vertical_speed,
					 }
				dict_list.append((self.time[k],self.time[k+1],argo_cfg))
			else:
				print('point outside domain, advancing')
		return dict_list

class AOMLFloat(FloatBase):
	max_depth = -10
	def __init__(self,argo_float,*args,**kwargs):
		self.time = TimeList(argo_float.prof.date)
		lats,lons = zip(*[(x.latitude,x.longitude) for x in argo_float.prof.pos])
		self.lons = LonList(lons)
		self.lats = LatList(lats)
		self.drift_depths = DepthList([-10]*(len(lons)-1))
		self.surface_time = 300 * 3600
		super().__init__(*args,**kwargs)

class ArgoFloat(FloatBase):
	max_depth = -2000
	def __init__(self,argo_float,*args,**kwargs):
		self.time = TimeList(argo_float.prof.date)
		lats,lons = zip(*[(x.latitude,x.longitude) for x in argo_float.prof.pos])
		self.lons = LonList(lons)
		self.lats = LatList(lats)
		try:
			if not argo_float.tech.drift_depth:	#if drift depth is empty
				self.drift_depths = DepthList([-1000]*(len(lons)-1))
			else:
				self.drift_depths = DepthList([-x for x in argo_float.tech.drift_depth._list])
		except AttributeError: #some argo files do not have tech files
			self.drift_depths = DepthList([-1000]*(len(lons)-1))
		if argo_float.meta.positioning_system == 'ARGOS':
			self.surface_time = 8 * 3600
		else:
			self.surface_time = .25 * 3600
		super().__init__(*args,**kwargs)



class HyperNavFloat(FloatBase):
	surface_time = 2 * 3600
	total_cycle_time = 1 * 86400
	def __init__(self,label,time,lons,lats,drift_depths,*args,**kwargs):
		self.time = TimeList([datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in time])	
		self.max_depth = drift_depths[0]
		self.drift_depths = drift_depths
		self.lats = lats
		self.lons = lons
		self.label = label
		super().__init__(*args,**kwargs)

def return_HI21A55():
	label='A55'
	time = ['2021-06-09 03:53:03','2021-06-09 19:01:19','2021-06-10 10:57:00','2021-06-11 07:39:10','2021-06-11 22:16:28',
	'2021-06-12 22:01:20','2021-06-13 21:52:14','2021-06-14 21:51:38','2021-06-15 21:47:54','2021-06-16 21:49:48','2021-06-17 21:47:43']
	lons = LonList([-156.4662,-156.4064,-156.3477,-156.2918,-156.2858,-156.2690,-156.3034,-156.3441,-156.4059,-156.4920,-156.5509])
	lats = LatList([19.5965, 19.5790, 19.5865, 19.6818, 19.7070, 19.7179, 19.7561, 19.7739, 19.7886, 19.7641, 19.6894])
	drift_depths = DepthList([-700]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)

def return_HI21A54():
	label='A54'
	time = ['2021-06-09 02:56:43','2021-06-09 17:36:19','2021-06-10 08:21:58','2021-06-10 23:49:05','2021-06-11 22:16:13','2021-06-12 21:55:34',
		'2021-06-13 21:52:06','2021-06-14 21:50:34','2021-06-15 21:55:20','2021-06-16 21:49:24']
	lons = LonList([-156.5018,-156.4330,-156.3792,-156.3354,-156.2685,-156.1731,-156.1575,-156.1579,-156.1597,-156.1958])
	lats = LatList([19.5942, 19.5698, 19.5605, 19.5571, 19.5399, 19.5416, 19.5577, 19.6003, 19.6558, 19.6924])
	drift_depths = DepthList([-700]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)


def return_HI21B55():
	label='B55'
	time = ['2021-06-18 01:19:00','2021-06-18 06:05:49','2021-06-18 21:45:54','2021-06-19 21:49:13','2021-06-20 21:49:30',
		'2021-06-21 21:48:56','2021-06-22 19:50:11']
	lons = LonList([-156.3901,-156.3991,-156.3896,-156.3452,-156.2445,-156.2115,-156.1658])
	lats = LatList([19.4567, 19.4448, 19.3951, 19.3190, 19.2701, 19.2193, 19.1824])
	drift_depths = DepthList([-700]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)


def return_HI21B54():
	label='B54'
	time = ['2021-06-18 02:39:40','2021-06-18 07:58:47','2021-06-18 22:02:34','2021-06-19 21:49:02','2021-06-20 21:50:44',
		'2021-06-21 21:46:52','2021-06-22 19:56:07']
	lons = LonList([-156.3957,-156.3975,-156.3920,-156.2536,-156.1781,-156.1714,-156.1906])
	lats = LatList([19.4518,19.4327,19.3894,19.2756,19.3127,19.2821,19.2079])
	drift_depths = DepthList([-700]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)

def return_HI21A53():
	label='A53'
	time = ['2021-11-16 21:00','2021-11-17 2:41','2021-11-17 18:13','2021-11-18 11:54','2021-11-19 3:14',
		'2021-11-19 22:37','2021-11-20 22:40','2021-11-21 22:39','2021-11-22 19:12']
	lons = LonList([-156.6906,-156.6516,-156.6161,-156.5401,-156.5244,-156.5333,-156.5501,-156.6043,-156.7921])
	lats = LatList([19.5998,19.6091,19.6663,19.7465,19.7959,19.8534,19.9313,20.0271,20.0888])
	drift_depths = DepthList([-700]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)


def return_HI22A53():
	label='B53'
	time = ['2022-04-04 23:38','2022-4-5 3:20','2022-04-05 10:04','2022-04-05 21:38','2022-4-6 2:40',
		'2022-4-6 8:21','2022-04-06 12:26','2022-04-06 20:44']
	lons = LonList([-156.4781,-156.3941,-156.2983,-156.154,-156.1263,-156.1115,-156.0995,-156.0633])
	lats = LatList([19.5436,19.5164,19.4662,19.3321,19.2745,19.2299,19.1828,19.0639])
	drift_depths = DepthList([-250]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)


def return_HI22B53():
	label='C53'
	time = ['2022-04-20 18:02','2022-04-20 21:07','2022-4-21 9:46',
	'2022-04-21 23:45','2022-04-22 21:59','2022-04-23 21:58','2022-04-24 21:59',
	'2022-04-25 22:01','2022-04-26 21:59','2022-04-27 21:59','2022-04-28 22:02',
	'2022-04-29 22:01','2022-04-30 21:56','2022-05-01 22:05','2022-05-02 22:27',
	'2022-05-03 22:05','2022-05-04 22:33','2022-05-05 21:59','2022-05-06 21:18',
	'2022-05-07 21:17','2022-05-08 21:19','2022-05-09 21:19','2022-05-10 21:17',
	'2022-05-11 22:04','2022-05-12 22:01']
	lons = LonList([-156.3107,-156.3394,-156.3889,-156.4461,-156.5239,-156.6219,
		-156.7313,-156.8758,-156.9602,-156.9922,-156.9863,-156.9688,-156.9481,-156.9119,
		-156.8644,-156.839,-156.8226,-156.7886,-156.7843,-156.7815,-156.7926,-156.8177,
		-156.846,-156.832,-156.7947])
	lats = LatList([19.6597,19.6996,19.7398,19.7757,19.8281,19.837,19.8417,
		19.7802,19.6529,19.5162,19.3782,19.2457,19.0038,18.9341,18.8568,18.8446,
		18.8317,18.8284,18.8751,18.9333,19.0536,19.175,19.3111,19.3892,19.4911])
	drift_depths = DepthList([-500]*18+[-250]*7)
	return HyperNavFloat(label,time,lons,lats,drift_depths)



def return_CR22A57():
	label='A57'
	time = ['2022-7-12 7:53','2022-07-12 12:10','2022-7-13 1:55','2022-07-13 15:46',
	'2022-7-14 5:41','2022-07-14 19:52','2022-7-15 9:04','2022-07-15 22:14',
	'2022-07-16 11:24','2022-7-17 0:33','2022-07-17 14:03','2022-7-18 3:13',
	'2022-07-18 16:25','2022-07-19 11:27','2022-07-20 11:26','2022-07-21 11:26',
	'2022-07-22 11:25','2022-07-23 11:22','2022-07-24 11:43','2022-07-25 12:04',
	'2022-07-26 11:45','2022-07-27 11:40','2022-7-28 7:08']
	lons = LonList([25.0696,25.0409,25.9865,25.9074,25.8379,25.7984,25.7736,
		25.7486,25.7471,25.7625,25.7828,25.7976,25.8064,25.8134,25.8042,25.7763,
		25.6767,25.6005,25.5912,25.6087,25.604,25.6063,25.6191])
	lats = LatList([35.739,35.7658,35.8301,35.8822,35.8986,35.8978,35.8949,
		35.8788,35.8542,35.8348,35.8217,35.8196,35.8207,35.8302,35.8508,
		35.9116,35.9847,35.9967,35.9922,36.011,36.0408,36.0717,36.1106])
	drift_depths = DepthList([-500]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)

def return_CR22A56():
	label='A56'
	time = ['2022-7-12 7:53','2022-07-12 12:10','2022-7-13 1:55','2022-07-13 15:46',
	'2022-7-14 5:41','2022-07-14 19:52','2022-7-15 9:04','2022-07-15 22:14',
	'2022-07-16 11:24','2022-7-17 0:33','2022-07-17 14:03','2022-7-18 3:13',
	'2022-07-18 16:25','2022-07-19 11:27','2022-07-20 11:26','2022-07-21 11:26',
	'2022-07-22 11:25','2022-07-23 11:22','2022-07-24 11:43','2022-07-25 12:04',
	'2022-07-26 11:45','2022-07-27 11:40','2022-7-28 7:08']
	lons = LonList([25.0696,25.0409,25.9865,25.9074,25.8379,25.7984,25.7736,
		25.7486,25.7471,25.7625,25.7828,25.7976,25.8064,25.8134,25.8042,25.7763,
		25.6767,25.6005,25.5912,25.6087,25.604,25.6063,25.6191])
	lats = LatList([35.739,35.7658,35.8301,35.8822,35.8986,35.8978,35.8949,
		35.8788,35.8542,35.8348,35.8217,35.8196,35.8207,35.8302,35.8508,
		35.9116,35.9847,35.9967,35.9922,36.011,36.0408,36.0717,36.1106])
	drift_depths = DepthList([-500]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)

def return_CR22A57():
	label='A57'
	time = ['2022-5-27 6:27','2022-5-27 9:55','2022-05-27 22:10',
	'2022-05-28 12:46','2022-05-29 10:22','2022-05-30 10:22','2022-5-31 8:55',
	'2022-6-1 8:55','2022-6-2 8:55','2022-6-3 8:56','2022-6-4 8:53',
	'2022-6-5 8:54','2022-6-6 8:54','2022-6-7 8:44','2022-6-8 8:45',
	'2022-6-9 8:37','2022-6-10 8:29','2022-6-11 8:31','2022-6-12 8:21',
	'2022-6-13 8:22','2022-6-14 8:21','2022-6-15 8:21','2022-6-16 8:29',
	'2022-6-17 8:21','2022-6-18 8:20','2022-6-19 8:29','2022-6-20 8:28',
	'2022-6-21 8:20','2022-6-22 8:28','2022-6-23 8:41','2022-6-24 8:23',
	'2022-6-25 8:29','2022-6-26 8:20','2022-6-27 8:30','2022-6-28 8:21',
	'2022-6-29 8:30','2022-6-30 8:30','2022-7-1 8:30','2022-7-2 8:31',
	'2022-7-3 8:23','2022-7-4 8:21','2022-7-5 8:21','2022-7-6 8:22',
	'2022-7-7 8:29','2022-7-8 8:21','2022-7-9 8:21','2022-7-10 8:25',
	'2022-7-11 8:22','2022-7-12 8:21','2022-7-13 8:59','2022-7-14 8:25',
	'2022-7-15 8:21','2022-7-16 8:21','2022-7-17 8:22','2022-7-18 8:22',
	'2022-7-19 8:22','2022-7-20 8:31','2022-7-21 8:25','2022-7-22 8:21',
	'2022-7-23 8:21','2022-7-24 8:21','2022-7-25 8:30','2022-7-26 8:22',
	'2022-7-27 8:21','2022-7-28 8:57','2022-7-29 8:21','2022-7-30 8:21',
	'2022-7-31 8:21','2022-8-1 5:17']

	lons = LonList([25.0671,25.0576,25.0323,25.0329,25.023,25.0661,25.1338,
		25.218,25.2846,25.2829,25.1688,25.0398,24.9915,25.0995,25.2883,25.4076,
		25.3023,25.1549,25.1243,25.2338,25.3505,25.2684,25.1324,25.1639,25.2888,
		25.3099,25.1936,25.16,25.2956,25.3614,25.2469,25.1818,25.2987,25.3706,
		25.218,25.1509,25.2324,25.3077,25.2021,25.1204,25.1221,25.2354,25.3772,
		25.3438,25.2078,25.0915,25.0882,25.1545,25.2567,25.1957,25.014,24.8898,
		24.8685,24.8977,24.9767,25.0992,25.1761,25.0447,24.8556,24.7331,24.7316,
		24.7669,24.811,24.8565,24.8904,24.8604,24.7732,24.7256,24.6987])

	lats = LatList([35.7406,35.7411,35.7816,35.8341,35.9661,36.0796,36.1038,
		36.0826,36.0181,36.8931,36.7902,35.8453,35.9978,36.1025,36.082,
		35.9052,35.7622,35.7698,35.8656,35.9532,35.8694,35.7646,35.8447,
		35.975,35.9695,35.8411,35.8347,35.9623,36.0306,35.9164,35.8584,
		35.9346,35.9922,35.8863,35.8579,35.9643,36.009,35.9122,35.8588,
		35.9514,36.0366,36.0738,35.9749,35.8144,35.7912,35.8949,36.0213,
		36.0659,35.9916,35.8655,35.9247,36.057,36.14,36.1855,36.1937,36.1411,
		36.0048,35.8973,35.9541,36.1308,36.2263,36.2661,36.2765,36.2452,
		36.1774,36.0671,36.0196,36.0455,36.1056])
	drift_depths = DepthList([-300]*(len(lons)-1))
	return HyperNavFloat(label,time,lons,lats,drift_depths)
