import pandas as pd
import numpy as np
import scipy
from GeneralUtilities.Data.Filepath.instance import get_data_folder 
import os
import matplotlib.pyplot as plt
from HyperNav.Utilities.Data.HYCOM import HYCOMPuertoRico,HYCOMMonterey
import geopy
import datetime
from scipy import signal
import math
from scipy import stats
import ttide as tt

def parse_df(df,subsample=1,days=16):
	freq_record = datetime.timedelta(days=days)
	df['diff_time'] = df.index - df.index[0]
	mod_list = []
	df['days'] = df.diff_time.apply(lambda x: x.days%freq_record.days)
	df['seconds'] = df.diff_time.apply(lambda x: x.seconds)

	df_list = np.split(df, np.flatnonzero(((df.days==0)&(df.seconds==0)).tolist()))[1:]
	good_list = []
	for temp_df in df_list:
		time_diff = temp_df.index.to_series().diff()
		if (~temp_df.u.isnull().values.any())&(time_diff.min()==time_diff.max())&((temp_df.index[-1]-temp_df.index[0]).days>=(days-1))&((temp_df.index[-1]-temp_df.index[0]+datetime.timedelta(hours=3)).days>=days):
			temp_df = temp_df.reset_index()
			temp_df['bin'] = temp_df.index //subsample
			temp_df = temp_df.groupby(['bin']).mean()
			good_list.append(temp_df)
	return good_list


class MooringBase():
	def __init__(self):
		u_std = self.hycom_df.u.std()
		v_std = self.hycom_df.v.std()
		self.hycom_df.loc[self.hycom_df.u>u_std*4]=3.5*u_std
		self.hycom_df.loc[self.hycom_df.u<-u_std*4]=-3.5*u_std
		self.hycom_df.loc[self.hycom_df.v>v_std*4]=3.5*v_std
		self.hycom_df.loc[self.hycom_df.v<-v_std*4]=-3.5*v_std

	def plot_uv_spectra_compare(self):

		def average_ffts(df,subsample=1):
			good_list = parse_df(df,subsample)
			fft_avg_list = []
			for temp_df in good_list:
				X = np.fft.fft(signal.detrend(temp_df.u.tolist()*signal.windows.hann(len(temp_df.u.tolist()))))
				fft_avg_list.append(X)
			X_avg = np.array(fft_avg_list).mean(axis=0)
			N = len(X_avg)
			freq = np.fft.fftfreq(N,3)
			return (X_avg, freq,N)


		X_mooring, freq_mooring,N_mooring = average_ffts(self.df,subsample=self.sampling_rate/8)
		X_hycom, freq_hycom,N_hycom = average_ffts(self.hycom_df)
		y_max = max(abs(X_mooring).tolist()+abs(X_hycom).tolist())

		fig = plt.figure()

		ax1 = fig.add_subplot(2, 1, 1)
		ax2 = fig.add_subplot(2, 2, 3, projection=self.uv.PlotClass.projection)
		ax3 = fig.add_subplot(2, 2, 4)

		alpha = 0.05

		err_low=N_mooring/stats.chi2.isf(1-alpha/2,N_mooring)
		err_high = N_mooring/stats.chi2.isf(alpha/2,N_mooring)

		x_index = round(len(freq_mooring)/2)

		ax1.plot(freq_mooring[0:x_index], abs(X_mooring[0:x_index]), 'b',label=self.name)
		ax1.fill_between(freq_mooring[0:x_index], err_high*abs(X_mooring[0:x_index]),err_low*abs(X_mooring[0:x_index]),color='b',alpha=0.2)

		ax1.set_yscale('log')
		ax1.set_xscale('log')
		ax1.set_xlabel('')
		ax1.set_xticks([])
		ax1.set_xlim(0, max(freq_hycom))
		ax1.set_ylim(0.001,y_max)

		err_low=N_hycom/stats.chi2.isf(1-alpha/2,N_hycom)
		err_high = N_hycom/stats.chi2.isf(alpha/2,N_hycom)

		x_index = round(len(freq_hycom)/2)


		ax1.plot(freq_mooring[0:x_index], abs(X_hycom[0:x_index]), 'r',label='HYCOM')
		ax1.fill_between(freq_mooring[0:x_index],err_low*abs(X_hycom[0:x_index]), err_high*abs(X_hycom[0:x_index]),color='r',alpha=0.2)
		# ax2.set_yscale('log')
		ax1.set_xscale('log')
		# ax1.set_xlim(0, max(freq_hycom))
		# ax1.set_ylim(0.001,y_max)
		ax1.legend()
		ax1.set_xticks([1/384.,1/192.,1/96.,1/48.,1/24.,2/24.,2/12.])
		ax1.set_xticklabels([r'$\frac{1}{16~days}$',r'$\frac{1}{8~days}$',r'$\frac{1}{4~days}$',r'$\frac{1}{2~days}$',r'$\frac{1}{day}$',r'$\frac{2}{day}$',r'$\frac{4}{day}$'])
		ax1.set_xlabel('Frequency')
		ax1.set_ylabel('FFT Amplitude |X(freq)|')

		XX,YY,ax2 = self.uv.plot(ax=ax2)
		ax2.scatter(self.location.longitude,self.location.latitude,s=100)

		self.df.u.plot(label=self.name,ax=ax3)
		self.hycom_df.u.plot(label='HYCOM',ax=ax3)
		ax3.legend()
		ax3.set_ylabel('Eastward Current Speed (m/s)')
		plt.show()

	def tidal_analysis(self):
		days = 40
		tide_holder = []
		
		for df,dt in [(self.df,24/self.sampling_rate),(self.hycom_df,3)]:
			print('######### one data frame #########')
			t_holder = []
			date_range = pd.date_range(df.index[0],df.index[-1],freq=datetime.timedelta(days=days))
			for k in range(len(date_range)-1):
				date_start = date_range[k]
				date_end = date_range[k+1]
				temp_df = df[(df.index>=date_start)&(df.index<=date_end)]
				time_diff = temp_df.index.to_series().diff()
				if temp_df.empty:
					continue
				if (~temp_df.u.isnull().values.any())&(time_diff.min()==time_diff.max())&((temp_df.index[-1]-temp_df.index[0]).days>=(days-1))&((temp_df.index[-1]-temp_df.index[0]+datetime.timedelta(hours=3)).days>=days):
					u = np.array(temp_df.u.tolist())
					v = np.array(temp_df.v.tolist())
					t_holder.append(tt(np.array([x + y*1j for x,y in zip(u,v)]),out_style=None,dt=dt))
					print(t_holder[-1]['tidecon'].shape)
					print(t_holder[-1]['fu'].shape)
			tide_holder.append((sum([x['tidecon'] for x in t_holder])/len(t_holder),t_holder[0]['fu']))

		fig = plt.figure(figsize=(30,5))
		ax1 = fig.add_subplot(2,1,1)
		ax1.plot(tide_holder[0][1],tide_holder[0][0][:,0],label=self.name)
		ax1.plot(tide_holder[1][1],tide_holder[1][0][:,0],label='HYCOM')
		ax1.set_yscale('log')
		nyquist = 1/(24/6.)
		truth_array = tide_holder[1][1]<nyquist
		ax1.set_xticks(ticks=tide_holder[1][1][truth_array],labels=['']*sum(truth_array),rotation=90)
		ax1.set_xlim(right=1/(7.))
		ax1.set_ylim(top=1)
		# ax1.set_xlabel('Frequency')
		ax1.set_ylabel('Amplitude')
		ax1.legend()

		ax2 = fig.add_subplot(2,1,2)
		# ax2.plot(tide_holder[0][1],tide_holder[0][0][:,6],label=self.name)
		# ax2.plot(tide_holder[1][1],tide_holder[1][0][:,6],label='HYCOM')
		phase_diff =tide_holder[0][0][:,6]-tide_holder[1][0][:,6]
		phase_diff = phase_diff%360
		phase_diff[phase_diff>180]=phase_diff[phase_diff>180]-360
		ax2.plot(tide_holder[0][1],phase_diff)
		ax2.plot(tide_holder[0][1],[0]*len(tide_holder[0][1]),'r--')

		# ax2.set_yscale('log')
		nyquist = 1/(24/6.)
		truth_array = tide_holder[1][1]<nyquist
		ax2.set_xticks(ticks=tide_holder[1][1][truth_array],labels=np.array([x.decode("utf-8") for x in t_holder[0]['nameu'].tolist()])[truth_array],rotation=90)
		ax2.set_xlim(right=1/(7.))
		ax2.set_ylim(-180,180)
		ax2.set_xlabel('Frequency')
		ax2.set_ylabel('Phase Difference (degrees)')
		ax2.legend()

		fig1 = plt.figure(figsize=(30,5))

		k1_index = [x.decode("utf-8") for x in t_holder[0]['nameu'].tolist()].index('K1  ')
		m2_index = [x.decode("utf-8") for x in t_holder[0]['nameu'].tolist()].index('M2  ')
		s2_index = [x.decode("utf-8") for x in t_holder[0]['nameu'].tolist()].index('S2  ')

		amp_concat_list = []
		phase_concat_list = []
		for idx,name in [(0,self.name),(1,self.name.replace('Mooring','HYCOM'))]:
			k1_amp = tide_holder[idx][0][k1_index,0]
			m2_amp = tide_holder[idx][0][m2_index,0]
			s2_amp = tide_holder[idx][0][s2_index,0]
			df_amp = pd.DataFrame({'K1':k1_amp,'S2':s2_amp,'M2':m2_amp},index=[name])
			amp_concat_list.append(df_amp)
		df_amp = pd.concat(amp_concat_list)


		k1_phase = phase_diff[k1_index]
		m2_phase = phase_diff[m2_index]
		s2_phase = phase_diff[s2_index]
		df_phase = pd.DataFrame({'K1':k1_phase,'S2':s2_phase,'M2':m2_phase},index=['Phase Difference'])


		ax3 = fig1.add_subplot(1,2,1)
		ax4 = fig1.add_subplot(1,2,2)

		df_amp.T.plot.bar(ax=ax3,ylabel='Amplitude')
		df_phase.T.plot.bar(ax=ax4,ylabel='Phase Difference (degrees)')
		plt.show()


class PuertoRicoMooring(MooringBase):
	uv = HYCOMPuertoRico.load(datetime.datetime(2016,1,1),datetime.datetime(2024,1,1),interpolate=False)
	def __init__(self):
		files = [filename for filename in os.listdir(self.path) if not filename.startswith('.')]
		holder = []
		for file in files:
			df_temp = pd.read_csv(os.path.join(self.path,file),sep='\s+',index_col=False)
			holder.append(df_temp)
		df = pd.concat(holder)
		df = df.rename(columns={'#YY':'year','MM':'month','DD':'day','hh':'hour','mm':'minute'})
		pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
		df = df.set_index(pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']]))
		df = df.drop(['year','month','day','hour','minute'],axis=1)
		df = df.sort_index()
		df.replace('MM', np.nan, inplace=True)

		dep_col_list = ['DEP{num:02d}'.format(num=x) for x in range(1,21)]
		dir_col_list = ['DIR{num:02d}'.format(num=x) for x in range(1,21)]
		spd_col_list = ['SPD{num:02d}'.format(num=x) for x in range(1,21)]

		master_pres_list = list(range(2,16))
		spd_array_list = []
		dir_array_list = []
		df = df.fillna(method='ffill',limit=2)
		for row in df.iterrows():
			print(row[0])
			pres_list = row[1][dep_col_list].astype(float).values
			dir_list = row[1][dir_col_list].astype(float).values
			spd_list = row[1][spd_col_list].astype(float).values

			interp_dir = np.interp(master_pres_list,pres_list,dir_list)
			dir_array_list.append(interp_dir)
			interp_spd = np.interp(master_pres_list,pres_list,spd_list)
			spd_array_list.append(interp_spd)
		spd_array = np.array(spd_array_list)
		spd_array = np.ma.masked_greater(spd_array,100)
		dir_array = np.array(dir_array_list)
		dir_array = np.ma.masked_greater(dir_array,360)
		v = np.cos(np.deg2rad(dir_array))*spd_array/100
		u = np.sin(np.deg2rad(dir_array))*spd_array/100


		lat_idx = self.uv.lats.find_nearest(self.location.latitude,idx=True)
		lon_idx = self.uv.lons.find_nearest(self.location.longitude,idx=True)
		hycom_u = self.uv.u[:,1,lat_idx,lon_idx]
		hycom_v = self.uv.v[:,1,lat_idx,lon_idx]


		self.df = pd.DataFrame({'u':u[:,0],'v':v[:,0]},index=df.index)
		self.hycom_df = pd.DataFrame({'u':hycom_u,'v':hycom_v},index=self.uv.time)
		super().__init__()



class PonceMooring(PuertoRicoMooring):
	path = os.path.join(get_data_folder(),'Raw/Moorings/42085')
	sampling_rate = 24. # per day
	location = geopy.Point(17.870,-66.527)
	name = 'Ponce Mooring'

class SanJuanMooring(PuertoRicoMooring):
	path = os.path.join(get_data_folder(),'Raw/Moorings/41053')
	sampling_rate = 24. # per day
	location = geopy.Point(18.474,-66.099)
	name = 'San Juan Mooring'


class StJohnMooring(PuertoRicoMooring):
	path = os.path.join(get_data_folder(),'Raw/Moorings/41052')
	sampling_rate = 24. # per day
	location = geopy.Point(18.249,-64.763)
	name = 'St John Mooring'


class ViequesMooring(PuertoRicoMooring):
	path = os.path.join(get_data_folder(),'Raw/Moorings/41056')
	sampling_rate = 24. # per day
	location = geopy.Point(18.261,-65.464)
	name = 'Vieques Mooring'

class MontereyMoorings(MooringBase):
	uv = HYCOMMonterey.load(datetime.datetime(2016,1,1),datetime.datetime(2024,1,1),interpolate=False)
	def __init__(self):
		super().__init__()


class M1Mooring(MontereyMoorings):
	path = os.path.join(get_data_folder(),'Raw/Moorings/M1')
	sampling_rate = 24. # per day
	location = geopy.Point(36.69623, -122.39965)
	name = 'M1 Mooring'
	def __init__(self):
		files = [filename for filename in os.listdir(self.path) if not filename.startswith('.')]
		holder = []
		for file in files:
			df_temp = pd.read_csv(os.path.join(self.path,file),index_col=False)
			holder.append(df_temp)
		df = pd.concat(holder)
		df['time'] = pd.to_datetime(df.time)
		df = df[(df.northward_sea_water_velocity_qc_agg==1)&(df.eastward_sea_water_velocity_qc_agg==1)]
		holder = []
		z_list = df.z.unique()
		for z in z_list:
			temp_df = df[df.z==z]
			u = temp_df.eastward_sea_water_velocity
			v = temp_df.northward_sea_water_velocity
			u_name = 'u_'+str(int(z))
			v_name = 'v_'+str(int(z))
			temp_df = pd.DataFrame({u_name:u,v_name:v,'time':temp_df.time})
			temp_df = temp_df.set_index('time',drop=True)
			holder.append(temp_df)
		df = pd.concat(holder,axis=1)
		df = df.sort_index()
		df = df.fillna(method='ffill',limit=2)
		df = df.dropna()


		lat_idx = self.uv.lats.find_nearest(self.location.latitude,idx=True)
		lon_idx = self.uv.lons.find_nearest(self.location.longitude,idx=True)
		hycom_u = self.uv.u[:,8,lat_idx,lon_idx]
		hycom_v = self.uv.v[:,8,lat_idx,lon_idx]


		self.df = pd.DataFrame({'u':df['u_-20'],'v':df['v_-20']},index=df.index)
		self.hycom_df = pd.DataFrame({'u':hycom_u,'v':hycom_v},index=self.uv.time)
		self.df = self.df.set_index(df.index.to_series().apply(lambda x: x.replace(minute=0, second=0)),drop=True)
		super().__init__()



class PointSurMooring(MontereyMoorings):
	path = os.path.join(get_data_folder(),'Raw/Moorings/46239')
	sampling_rate = 8. # per day
	location = geopy.Point(36.3347,-122.1039)
	name = 'Point Sur Mooring'
	def __init__(self):
		files = [filename for filename in os.listdir(self.path) if not filename.startswith('.')]
		holder = []
		for file in files:
			df_temp = pd.read_csv(os.path.join(self.path,file),sep='\s+',index_col=False)
			holder.append(df_temp)
		df = pd.concat(holder)
		df = df.rename(columns={'#YY':'year','MM':'month','DD':'day','hh':'hour','mm':'minute'})
		pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
		df = df.set_index(pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']]))
		df = df.drop(['year','month','day','hour','minute'],axis=1)
		df = df.sort_index()
		df.replace('MM', np.nan, inplace=True)

		dep_col_list = ['DEP{num:02d}'.format(num=x) for x in range(1,21)]
		dir_col_list = ['DIR{num:02d}'.format(num=x) for x in range(1,21)]
		spd_col_list = ['SPD{num:02d}'.format(num=x) for x in range(1,21)]

		master_pres_list = list(range(2,16))
		spd_array_list = []
		dir_array_list = []
		df = df.fillna(method='ffill',limit=2)

		df = df[df['DIR01']<361]
		dir_array = df.DIR01.tolist()
		spd_array = df.SPD01.tolist()

		v = np.cos(np.deg2rad(dir_array))*spd_array/100
		u = np.sin(np.deg2rad(dir_array))*spd_array/100


		lat_idx = self.uv.lats.find_nearest(self.location.latitude,idx=True)
		lon_idx = self.uv.lons.find_nearest(self.location.longitude,idx=True)
		hycom_u = self.uv.u[:,1,lat_idx,lon_idx]
		hycom_v = self.uv.v[:,1,lat_idx,lon_idx]

		self.df = pd.DataFrame({'u':u,'v':v},index=df.index)
		self.df['bin'] = self.df.index.to_series().apply(lambda x: (x.year,x.month,x.day,math.floor(x.hour/3)))
		self.df = self.df.groupby(['bin']).mean()
		self.df = self.df.set_index(self.df.index.to_series().apply(lambda x: datetime.datetime(x[0],x[1],x[2],x[3]*3)))
		self.df = self.df.reindex(pd.date_range(start=self.df.index[0],end=self.df.index[-1],freq=datetime.timedelta(hours=3)))
		self.df = self.df.interpolate(method='ffill')
		self.hycom_df = pd.DataFrame({'u':hycom_u,'v':hycom_v},index=self.uv.time)
		super().__init__()


mooring_list = []
for mooring in [PonceMooring,SanJuanMooring,ViequesMooring,M1Mooring,PointSurMooring]:
	mooring_list.append(mooring())
