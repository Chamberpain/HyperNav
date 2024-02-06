from HyperNav.Utilities.Data.SVC import SVCBermuda,SVCSoCal,SVCCrete,SVCHawaii,SVCMonterey,SVCPuertoRico,SVCTahiti
from HyperNav.Utilities.Data.ClearSky import ClearSkyBermuda,ClearSkySoCal,ClearSkyCrete,ClearSkyHawaii,ClearSkyMonterey,ClearSkyPuertoRico,ClearSkyTahiti
import geopy
import datetime
import numpy as np
import matplotlib.pyplot as plt

class FinanceBase():
	hardware = 225000/8
	transmission = (31104+1575)/3
	site_fees = (6000+6000)/3
	recalibration = 0
	fixed_cost = hardware+transmission+site_fees+recalibration

	def matchup_num(self,month,matchup_class):
		num_list = []
		for dummy in range(200):
			match_up_holder = []
			date = datetime.date(2022,month,1)
			for day in range(30):
				date += datetime.timedelta(days=1)
				match_up_holder.append(matchup_class(self.deployed_point.latitude,self.deployed_point.longitude,date.month))
			num_list.append(sum(match_up_holder))
		return (np.mean(num_list),np.std(num_list))

	def price(self):
		return self.fixed_cost+self.travel+self.logistics+self.boat

	def price_per_matchup(self,month,sky_type):
		if sky_type=='SVC':
			print(self.matchup_num(month,self.svcclass.return_svc_matchup)[1]/self.matchup_num(month,self.svcclass.return_svc_matchup)[0])
			return (self.price()/(self.matchup_num(month,self.svcclass.return_svc_matchup)[0]),self.price()/self.matchup_num(month,self.svcclass.return_svc_matchup)[0]*(self.matchup_num(month,self.svcclass.return_svc_matchup)[1])/self.matchup_num(month,self.svcclass.return_svc_matchup)[0])
		if sky_type=='ClearSky':
			print(self.matchup_num(month,self.skyclass.match_up)[1]/self.matchup_num(month,self.svcclass.return_svc_matchup)[0])
			return (self.price()/(self.matchup_num(month,self.skyclass.match_up)[0]),self.price()/self.matchup_num(month,self.svcclass.return_svc_matchup)[0]*(self.matchup_num(month,self.skyclass.match_up)[1])/self.matchup_num(month,self.svcclass.return_svc_matchup)[0])
		if sky_type=='Zibordi':
			print(self.matchup_num(month,self.svcclass.return_svc_matchup_zibordi)[1]/self.matchup_num(month,self.svcclass.return_svc_matchup_zibordi)[0])
			return (self.price()/(self.matchup_num(month,self.svcclass.return_svc_matchup_zibordi)[0]),self.price()/self.matchup_num(month,self.svcclass.return_svc_matchup)[0]*(self.matchup_num(month,self.svcclass.return_svc_matchup_zibordi)[1])/self.matchup_num(month,self.svcclass.return_svc_matchup_zibordi)[0])



	def price_per_month(self,sky_type):
		month_price = []
		month_std = []
		for month in range(1,13):
			month_price.append(self.price_per_matchup(month,sky_type)[0])
			month_std.append(self.price_per_matchup(month,sky_type)[1])
		return (month_price,month_std)

	def matchup_per_month(self,sky_type):
		month_match = []
		month_std = []
		for month in range(1,13):
			if sky_type=='SVC':
				month_match.append(self.matchup_num(month,self.svcclass.return_svc_matchup)[0])
				month_std.append(self.matchup_num(month,self.svcclass.return_svc_matchup)[1])
			if sky_type=='ClearSky':
				month_match.append(self.matchup_num(month,self.skyclass.match_up)[0])
				month_std.append(self.matchup_num(month,self.skyclass.match_up)[1])
			if sky_type=='Zibordi':
				month_match.append(self.matchup_num(month,self.svcclass.return_svc_matchup_zibordi)[0])
				month_std.append(self.matchup_num(month,self.svcclass.return_svc_matchup_zibordi)[1])
		return (month_match,month_std)

class MontereyFinance(FinanceBase):
	label = 'Monterey'
	deployed_point = geopy.Point(36.7,-122.2)
	svcclass = SVCMonterey()
	skyclass = ClearSkyMonterey()
	lodging = 184
	meals = 69
	flight = 238
	logistics = 4500
	boat = 1500*3
	travel = (flight*4+lodging*8+meals*8)/0.49

class SoCalFinance(FinanceBase):
	label = 'Port Hueneme'
	deployed_point = geopy.Point(32.5,-117.5)
	svcclass = SVCSoCal()
	skyclass = ClearSkySoCal()
	travel = 1000
	logistics = 480*2
	boat = 900*2

class HawaiiFinance(FinanceBase):
	label = 'Kona'
	deployed_point = geopy.Point(19.5,-156.3)
	svcclass = SVCHawaii()
	skyclass = ClearSkyHawaii()
	lodging = 229
	meals = 138
	flight = 598
	logistics = 1000*2
	boat = 2088*4
	travel = (flight*4+lodging*8+meals*8)/0.49

class MobyFinance(FinanceBase):
	label = 'MOBY'
	deployed_point = geopy.Point(20.8,-157.2)
	svcclass = SVCHawaii()
	skyclass = ClearSkyHawaii()
	travel = 15146*2
	logistics = 9070
	boat = 2088*2

class CreteFinance(FinanceBase):
	label = 'Crete'
	deployed_point = geopy.Point(35.75,25.0)
	svcclass = SVCCrete()
	skyclass = ClearSkyCrete()
	lodging = 0
	meals = 0
	flight = 0
	logistics = 2585*2+200 # for carnet
	boat = 2500*2
	travel = (flight*4+lodging*8+meals*8)/0.49

class PRFinance(FinanceBase):
	label = 'Puerto Rico'
	deployed_point = geopy.Point(17.5,-66.5)
	svcclass = SVCPuertoRico()
	skyclass = ClearSkyPuertoRico()
	flight = 516
	lodging = 109
	meals = 75
	logistics = 900*2
	boat = 1300*3
	travel = (flight*4+lodging*8+meals*8)/0.49


class TahitiFinance(FinanceBase):
	label = 'Tahiti'
	deployed_point = geopy.Point(-17.8,-149.75)
	svcclass = SVCTahiti()
	skyclass = ClearSkyTahiti()
	lodging = 109
	flight = 1200
	meals = 124
	travel = (flight*4+lodging*8+meals*8)/0.49
	logistics = 2585*2
	boat = 2500*2

class BermudaFinance(FinanceBase):
	label = 'Bermuda'
	deployed_point = geopy.Point(32,-64.5)
	svcclass = SVCBermuda()
	skyclass = ClearSkyBermuda()
	lodging = 385
	flight = 1200
	meals = 158
	travel = (flight*4+lodging*8+meals*8)/0.49
	logistics = 2585*2
	boat = 2500*3

class CanaryFinance(FinanceBase):
	label = 'Carnary'
	deployed_point = geopy.Point(27.75,-16.5)
	# svcclass = SVCCanary()
	# skyclass = ClearSkyCanary()
	lodging = 179
	flight = 2200
	meals = 71
	travel = (flight*4+lodging*8+meals*8)/0.49
	logistics = 2585*2
	boat = 5000

def cost_plot():
	CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
	fig, (ax1, ax2, ax3) = plt.subplots(3)
	month_dummy = range(12)
	for k,(costclass,color) in enumerate(zip([HawaiiFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance],CB_color_cycle[:6])):
		num,std = costclass().price_per_month('SVC')
		print(costclass.label)
		print(num)
		ax1.plot(month_dummy,num,label=costclass.label,color=color)
	ax1.set_xticks([0,2,4,6,8,10],['','','','','',''])
	ax1.set_ylabel('Cost/Matchup ($)')
	ax1.set_yscale('log')
	ax1.set_ylim([10000,350000])
	ax1.set_xlim(0,11)
	ax1.annotate('a', xy = (0.1,0.85),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	for k,(costclass,color) in enumerate(zip([HawaiiFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance],CB_color_cycle[:6])):
		num,std = costclass().price_per_month('Zibordi')
		ax2.plot(month_dummy,num,label=costclass.label,color=color)	
	ax2.set_xticks([0,2,4,6,8,10],['','','','','',''])
	ax2.set_ylabel('Cost/Matchup ($)')
	ax2.set_yscale('log')
	ax2.set_ylim([10000,350000])
	ax2.set_xlim(0,11)
	ax2.annotate('b', xy = (0.1,0.85),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	for k,(costclass,color) in enumerate(zip([HawaiiFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance,MontereyFinance,SoCalFinance],CB_color_cycle[:8])):
		num,std = costclass().price_per_month('ClearSky')
		ax3.plot(month_dummy,num,label=costclass.label,color=color)
	ax3.set_ylabel('Cost/Matchup ($)')
	ax3.set_yscale('log')
	ax3.set_ylim([3000,60000])
	ax3.set_xlim(0,11)
	ax3.annotate('c', xy = (0.1,0.85),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	ax3.legend(loc='upper center', ncol=4,bbox_to_anchor=(0.5, 4))
	ax3.set_xticks([0,2,4,6,8,10],['Jan','Mar','May','Jul','Sep','Nov'])
	ax3.set_xlabel('Month')
	plt.show()

def operations_plot():
	fig, ax = plt.subplots()
	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,MobyFinance,PRFinance,CreteFinance,PRFinance,TahitiFinance,BermudaFinance]:
		price = costclass().price_per_month()
		if costclass.label in ['Monterey','Puerto Rico','Tahiti']:
			ax.plot(price,label=costclass.label,alpha=0.4)			
		else:
			ax.plot(price,linewidth=4,label=costclass.label)
	plt.legend()
	ax.set_xticklabels(['','Jan','Mar','May','Jul','Sep','Nov'])
	ax.set_xlabel('Month')
	ax.set_ylabel('Cost/Matchup ($)')
	ax.set_yscale('log')
	plt.show()



def match_per_month():
	num_list = list(zip(SoCalFinance().matchup_per_month(),HawaiiFinance().matchup_per_month()
		,CreteFinance().matchup_per_month()))


def num_plot():
	CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
	fig, (ax1, ax2, ax3) = plt.subplots(3)
	month_dummy = range(12)
	for k,(costclass,color) in enumerate(zip([HawaiiFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance,MontereyFinance,SoCalFinance],CB_color_cycle[:6])):
		num,std = costclass().matchup_per_month('SVC')
		print(costclass.label)
		print(num)
		ax1.errorbar(np.array(month_dummy)-0.1+k*0.03,num,ls='none',yerr=std,color=color,alpha=0.7)
		ax1.plot(month_dummy,num,label=costclass.label,color=color)
	ax1.set_xticks([0,2,4,6,8,10],['','','','','',''])
	ax1.set_ylabel('Matchup Number')
	ax1.set_xlim(0,11)
	ax1.annotate('a', xy = (-0.1,0.85),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	for k,(costclass,color) in enumerate(zip([HawaiiFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance,MontereyFinance,SoCalFinance],CB_color_cycle[:6])):
		num,std = costclass().matchup_per_month('Zibordi')
		ax2.errorbar(np.array(month_dummy)-0.1+k*0.03,num,ls='none',yerr=std,color=color,alpha=0.7)
		ax2.plot(month_dummy,num,label=costclass.label,color=color)	
	ax2.set_xticks([0,2,4,6,8,10],['','','','','',''])
	ax2.set_ylabel('Matchup Number')
	ax2.set_xlim(0,11)
	ax2.annotate('b', xy = (-0.1,0.85),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	for k,(costclass,color) in enumerate(zip([HawaiiFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance,MontereyFinance,SoCalFinance],CB_color_cycle[:8])):
		num,std = costclass().matchup_per_month('ClearSky')
		ax3.errorbar(np.array(month_dummy)-0.1+k*0.03,num,ls='none',yerr=std,color=color,alpha=0.7)
		ax3.plot(month_dummy,num,label=costclass.label,color=color)
	ax3.set_xticks([0,2,4,6,8,10],['','','','','',''])
	ax3.set_ylabel('Matchup Number')
	ax3.set_xlim(0,11)
	ax3.annotate('c', xy = (-0.1,0.85),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	ax3.legend(loc='upper center', ncol=4,bbox_to_anchor=(0.5, 4))
	ax3.set_xticks([0,2,4,6,8,10],['Jan','Mar','May','Jul','Sep','Nov'])
	ax3.set_xlabel('Month')
	plt.show()


def zibordi_frouin_num_plot():
	CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

	fig, (ax1, ax2,ax4) = plt.subplots(3)
	fig1, ax3 = plt.subplots(1)

	for costclass,color in zip([HawaiiFinance,MobyFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance],CB_color_cycle):
		Frouin_Num,Frouin_std = costclass().matchup_per_month('SVC')
		ax1.plot(np.array(Frouin_Num),label=costclass.label,color=color)
		# ax1.fill_between(range(len(Frouin_std)),np.array(Frouin_Num)-np.array(Frouin_std),np.array(Frouin_Num)+np.array(Frouin_std),color=color,alpha=0.3)
		Zibordi_Num,Zibordi_std = costclass().matchup_per_month('Zibordi')
		ax2.plot(Zibordi_Num,label=costclass.label,color=color)		
		# ax2.fill_between(range(len(Zibordi_std)),np.array(Zibordi_Num)-np.array(Zibordi_std),np.array(Zibordi_Num)+np.array(Zibordi_std),color=color,alpha=0.3)

		ax3.plot(np.array(np.array(Frouin_Num)-np.array(Zibordi_Num)),label=costclass.label,color=color)
		Clearsky_Num,Clearsky_std = costclass().matchup_per_month('ClearSky')
		ax4.plot(Clearsky_Num,label=costclass.label,color=color)
		# ax4.fill_between(range(len(Clearsky_std)),np.array(Clearsky_Num)-np.array(Clearsky_std),np.array(Clearsky_Num)+np.array(Clearsky_std),color=color,alpha=0.3)

	for costclass,color in zip([MontereyFinance,SoCalFinance],CB_color_cycle[-2:]):
		Clearsky_Num,ClearSky_std = costclass().matchup_per_month('ClearSky')
		ax4.plot(Clearsky_Num,label=costclass.label,color=color)
		# ax4.fill_between(range(len(Clearsky_std)),np.array(Clearsky_Num)-np.array(Clearsky_std),np.array(Clearsky_Num)+np.array(Clearsky_std),color=color,alpha=0.3)

	ax1.set_xticklabels(['','','','','','','',])
	ax1.set_xlabel('')
	ax1.set_ylabel('Matchup Number')
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	ax1.set_ylim(0,5)
	ax2.set_xticklabels(['','','','','','','',])
	ax2.set_xlabel('')
	ax2.set_ylabel('Matchup Number')
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.set_ylim(0,5)
	ax4.set_xticklabels(['','Jan','Mar','May','Jul','Sep','Nov'])
	ax4.set_xlabel('Month')
	ax4.set_ylabel('Matchup Number')
	ax4.annotate('c', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	ax4.set_ylim(0,12)
	ax4.legend(loc='upper center', ncol=4,bbox_to_anchor=(0.5, 4))

	ax3.legend(loc='upper center', ncol=3,bbox_to_anchor=(0.5, 1.15))
	ax3.set_xticklabels(['','Jan','Mar','May','Jul','Sep','Nov'])	
	ax3.set_xlabel('Month')
	ax3.set_ylabel('Frouin - Zibordi Matchup Number')	
	plt.show()

def bar_plot():
	fig, ax = plt.subplots()
	barWidth = 0.25
	label = []
	# hardware = []
	# transmission = []
	# site_fees = []
	# recalibration = []
	travel = []
	logistics = []
	boat = []

	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,CreteFinance,PRFinance,TahitiFinance]:
		label.append(costclass.label)
		# hardware.append(costclass.hardware)
		# transmission.append(costclass.transmission)
		# site_fees.append(costclass.site_fees)
		# recalibration.append(costclass.recalibration)
		travel.append(costclass.travel)
		logistics.append(costclass.logistics)
		boat.append(costclass.boat)

	br_hardware = np.arange(len(hardware))
	br_transmission = [x + barWidth for x in br_hardware]
	br_site = [x + barWidth for x in br_transmission]
	br_recalibration = [x + barWidth for x in br_site]
	br_travel = [x + barWidth for x in br_recalibration]
	br_logistics = [x + barWidth for x in br_travel]
	br_boat = [x + barWidth for x in br_logistics]

	# plt.bar(br_hardware, hardware, color ='blue', width = barWidth,
	#         edgecolor ='grey', label ='Hardware')
	# plt.bar(br_transmission, transmission, color ='orange', width = barWidth,
	#         edgecolor ='grey', label ='Transmission')
	# plt.bar(br_site, site_fees, color ='green', width = barWidth,
	#         edgecolor ='grey', label ='Site')
	# plt.bar(br_recalibration, recalibration, color ='purple', width = barWidth,
	#         edgecolor ='grey', label ='Recalibration')
	plt.bar(br_travel, travel, color ='brown', width = barWidth,
	        edgecolor ='grey', label ='Travel')
	plt.bar(br_logistics, logistics, color ='pink', width = barWidth,
	        edgecolor ='grey', label ='Logistics')
	plt.bar(br_boat, boat, color ='olive', width = barWidth,
	        edgecolor ='grey', label ='Boat')

	ax.set_xticklabels(['']+label)
	ax.set_xlabel('Site')
	ax.set_ylabel('Cost ($)')
	plt.legend()
	plt.show()
