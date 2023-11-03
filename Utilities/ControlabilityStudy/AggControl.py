from HyperNav.Utilities.ControlabilityStudy.ControlBase import ControlMonterey,ControlSoCal,ControlHawaii,ControlPR,ControlBermuda,ControlTahiti,ControlCrete
import os
import numpy as np
import matplotlib.pyplot as plt

class AggregateControlBase():
	def __init__(self):
		self.class_list = []
		for run in range(100):
			filename = self.control.make_pickle_filename(run,0)
			if os.path.exists(filename):
				self.class_list.append(self.control(run,0,0))

	def return_dist_stats(self):
		mean_list = []
		std_list = []
		for idx in range(self.control.profile_num):
			dist_list = [x.dist[idx] if idx<len(x.dist) else x.dist[-1] for x in self.class_list]
			dist_list = [x for x in dist_list if x is not None]
			mean_list.append(np.mean(dist_list))
			std_list.append(np.std(dist_list))
		return (mean_list,std_list)

class AggregateMonterey(AggregateControlBase):
	control = ControlMonterey

class AggregateBermuda(AggregateControlBase):
	control = ControlBermuda

class AggregateSoCal(AggregateControlBase):
	control = ControlSoCal

class AggregateHawaii(AggregateControlBase):
	control = ControlHawaii

class AggregatePR(AggregateControlBase):
	control = ControlPR

class AggregateCrete(AggregateControlBase):
	control = ControlCrete

class AggregateTahiti(AggregateControlBase):
	control = ControlTahiti

def plot_distance_stats():
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(1,1,1)
	name_dict = {'PuertoRico':'Puerto Rico','Hawaii':'Hawaii',
	'SoCal':'Port Hueneme','Monterey':'Monterey','Bermuda':'Bermuda','Crete':'Crete','Tahiti':'Tahiti'}
	
	for aggclass in [AggregatePR,AggregateHawaii,AggregateSoCal,AggregateMonterey,AggregateBermuda,AggregateCrete,AggregateTahiti]:
		class_holder = aggclass()
		mean_holder, std_holder = class_holder.return_dist_stats()
		mean_holder = np.array(mean_holder)
		std_holder = np.array(std_holder)
		x = [x for x in range(len(mean_holder))]
		name = name_dict[aggclass.control.uv_class.location]
		print(name)
		print(max(mean_holder)/max(std_holder))
		ax.plot(x,mean_holder,label=name)
		ax.fill_between(x,mean_holder-std_holder,mean_holder+std_holder,alpha=0.2)
	plt.ylim(ymin=0)
	plt.xlim(xmin=0,xmax=60)
	plt.ylabel('Distance From Deployment (km)')
	plt.xlabel('Profile Number')
	plt.legend()
	plt.show()