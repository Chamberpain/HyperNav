from HyperNav.Utilities.Data.ClearSky import ClearSkySoCal,ClearSkyCrete,ClearSkyHawaii,ClearSkyMonterey,ClearSkyPuertoRico,ClearSkyTahiti
import geopy
import datetime
import numpy as np
import matplotlib.pyplot as plt

class FinanceBase():
	hardware = 225000/2
	transmission = (31104+1575)/3
	site_fees = (6000+6000)/3
	recalibration = 0
	fixed_cost = hardware+transmission+site_fees+recalibration

	def matchup_num(self,month):
		num_list = []
		for dummy in range(10):
			match_up_holder = []
			date = datetime.date(2022,month,1)
			for day in range(60):
				date += datetime.timedelta(days=1)
				match_up_holder.append(self.skyclass.return_clear_sky(self.deployed_point.latitude,self.deployed_point.longitude,date.month))
			num_list.append(sum(match_up_holder))
		return np.mean(num_list)

	def price(self):
		return self.fixed_cost+self.travel+self.logistics+self.boat

	def price_per_matchup(self,month):
		return self.price()/(self.matchup_num(month))

	def price_per_month(self):
		month_price = []
		for month in range(1,13):
			month_price.append(self.price_per_matchup(month))
		return month_price

	def matchup_per_month(self):
		month_match = []
		for month in range(1,13):
			month_match.append(self.matchup_num(month))
		return month_match

class MontereyFinance(FinanceBase):
	label = 'Monterey'
	deployed_point = geopy.Point(36.7,-122.2)
	skyclass = ClearSkyMonterey()
	travel = 4500*2
	logistics = 4500
	boat = 2088*2

class SoCalFinance(FinanceBase):
	label = 'San Diego'
	deployed_point = geopy.Point(32.5,-117.5)
	skyclass = ClearSkySoCal()
	travel = 4000*2
	logistics = 4500
	boat = 2088*2

class HawaiiFinance(FinanceBase):
	label = 'Hawaii'
	deployed_point = geopy.Point(19.5,-156.3)
	skyclass = ClearSkyHawaii()
	travel = 15146*2
	logistics = 9070
	boat = 2088*2

class CreteFinance(FinanceBase):
	label = 'Crete'
	deployed_point = geopy.Point(35.75,25.0)
	skyclass = ClearSkyCrete()
	travel = 20218*2
	logistics = 9070
	boat = 5000

class PRFinance(FinanceBase):
	label = 'Puerto Rico'
	deployed_point = geopy.Point(17.5,-66.5)
	skyclass = ClearSkyPuertoRico()
	travel = 10856*2
	logistics = 9070
	boat = 10444


class TahitiFinance(FinanceBase):
	label = 'Tahiti'
	deployed_point = geopy.Point(17.8,-149.75)
	skyclass = ClearSkyTahiti()
	travel = 20218*2
	logistics = 9070
	boat = 5000


def cost_plot():
	fig, ax = plt.subplots()
	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,CreteFinance,PRFinance,TahitiFinance]:
		price = costclass().price_per_month()
		ax.plot(price,label=costclass.label)
	plt.legend()
	ax.set_xticklabels(['','Jan','Mar','May','Jul','Sep','Nov'])
	ax.set_xlabel('Month')
	ax.set_ylabel('Cost/Matchup ($)')
	ax.set_yscale('log')
	plt.show()

def operations_plot():
	fig, ax = plt.subplots()
	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,CreteFinance,PRFinance,TahitiFinance]:
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
	fig, ax = plt.subplots()
	for costclass in [MontereyFinance,SoCalFinance,HawaiiFinance,CreteFinance,PRFinance,TahitiFinance]:
		num = costclass().matchup_per_month()
		ax.plot(np.array(num)/2,label=costclass.label)
	plt.legend(loc=4)
	ax.set_xticklabels(['','Jan','Mar','May','Jul','Sep','Nov'])
	ax.set_xlabel('Month')
	ax.set_ylabel('Matchup Number')
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
