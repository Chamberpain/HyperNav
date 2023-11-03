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
		for dummy in range(10):
			match_up_holder = []
			date = datetime.date(2022,month,1)
			for day in range(30):
				date += datetime.timedelta(days=1)
				match_up_holder.append(matchup_class(self.deployed_point.latitude,self.deployed_point.longitude,date.month))
			num_list.append(sum(match_up_holder))
		return np.mean(num_list)

	def price(self):
		return self.fixed_cost+self.travel+self.logistics+self.boat

	def price_per_matchup(self,month,sky_type):
		if sky_type=='SVC':
			return self.price()/(2*self.matchup_num(month,self.svcclass.return_svc_matchup))
		if sky_type=='ClearSky':
			return self.price()/(2*self.matchup_num(month,self.skyclass.match_up))

	def price_per_month(self,sky_type):
		month_price = []
		for month in range(1,13):
			month_price.append(self.price_per_matchup(month,sky_type))
		return month_price

	def matchup_per_month(self,sky_type):
		month_match = []
		for month in range(1,13):
			if sky_type=='SVC':
				month_match.append(self.matchup_num(month,self.svcclass.return_svc_matchup))
			if sky_type=='ClearSky':
				month_match.append(self.matchup_num(month,self.skyclass.match_up))
		return month_match

class MontereyFinance(FinanceBase):
	label = 'Monterey'
	deployed_point = geopy.Point(36.7,-122.2)
	svcclass = SVCMonterey()
	skyclass = ClearSkyMonterey()
	travel = 4500*2
	logistics = 4500
	boat = 1500*3

class SoCalFinance(FinanceBase):
	label = 'Port Hueneme'
	deployed_point = geopy.Point(32.5,-117.5)
	svcclass = SVCSoCal()
	skyclass = ClearSkySoCal()
	travel = 1000
	logistics = 4500
	boat = 900*2

class HawaiiFinance(FinanceBase):
	label = 'Kona'
	deployed_point = geopy.Point(19.5,-156.3)
	svcclass = SVCHawaii()
	skyclass = ClearSkyHawaii()
	travel = 15146
	logistics = 9070
	boat = 2088*4

class MobyFinance(FinanceBase):
	label = 'Moby'
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
	travel = 20218
	logistics = 9070
	boat = 2500*2

class PRFinance(FinanceBase):
	label = 'Puerto Rico'
	deployed_point = geopy.Point(17.5,-66.5)
	svcclass = SVCPuertoRico()
	skyclass = ClearSkyPuertoRico()
	travel = 10856*2
	logistics = 9070
	boat = 2500*3


class TahitiFinance(FinanceBase):
	label = 'Tahiti'
	deployed_point = geopy.Point(-17.8,-149.75)
	svcclass = SVCTahiti()
	skyclass = ClearSkyTahiti()
	travel = 20218
	logistics = 9070
	boat = 2500*2

class BermudaFinance(FinanceBase):
	label = 'Bermuda'
	deployed_point = geopy.Point(32,-64.5)
	svcclass = SVCBermuda()
	skyclass = ClearSkyBermuda()
	travel = 20218
	logistics = 9070
	boat = 2500*3

class CanaryFinance(FinanceBase):
	label = 'Carnary'
	deployed_point = geopy.Point(27.75,-16.5)
	# svcclass = SVCCanary()
	# skyclass = ClearSkyCanary()
	travel = 20218*2
	logistics = 9070
	boat = 5000

def cost_plot():
	fig, (ax1, ax2) = plt.subplots(2)
	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance]:
		price = costclass().price_per_month('SVC')
		print(costclass.label)
		print(price)
		ax1.plot(price,label=costclass.label)
	ax1.set_xticks([0,2,4,6,8,10],['','','','','',''])
	ax1.set_ylabel('Cost/Matchup ($)')
	ax1.set_yscale('log')
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance]:
		price = costclass().price_per_month('ClearSky')
		ax2.plot(price,label=costclass.label)
	ax1.legend(loc='upper center', ncol=4,bbox_to_anchor=(0.5, 1.3))
	ax2.set_xticks([0,2,4,6,8,10],['Jan','Mar','May','Jul','Sep','Nov'])
	ax2.set_xlabel('Month')
	ax2.set_ylabel('Cost/Matchup ($)')
	ax2.set_yscale('log')
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
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
	fig, (ax1, ax2) = plt.subplots(2)
	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,MobyFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance]:
		num = costclass().matchup_per_month('SVC')
		ax1.plot(np.array(num),label=costclass.label)
	ax1.set_xticklabels(['','','','','','','',])
	ax1.set_xlabel('')
	ax1.set_ylabel('Matchup Number')
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,MobyFinance,PRFinance,CreteFinance,TahitiFinance,BermudaFinance]:
		price = costclass().matchup_per_month('ClearSky')
		ax2.plot(price,label=costclass.label)
	ax1.legend(loc='upper center', ncol=4,bbox_to_anchor=(0.5, 1.3))
	ax2.set_xticklabels(['','Jan','Mar','May','Jul','Sep','Nov'])
	ax2.set_xlabel('Month')
	ax2.set_ylabel('Matchup Number')
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=20,bbox=dict(boxstyle="round", fc="0.8"),)

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
