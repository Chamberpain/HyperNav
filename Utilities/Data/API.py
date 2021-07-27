import requests

class SiteAPI():
	api_url = 'http://misclab.umeoce.maine.edu/HyperNAV/api/v1/'
	access_token = 'DRehY4yxIFog2o8BZaNk752TqFgN3hHG8yRsB1uGfImwbXmJ8jM1ENQxsvJDCiOZJGNpeo4AL7bKQXVcLWDUV6rSEusEq'

	@staticmethod
	def upload_prediction(predictions,ID):
		response = requests.put(SiteAPI.api_url + 'position_predictions/' + ID,
								headers={'Authorization': SiteAPI.access_token},
								json=predictions)
		if response.status_code == 200:
			print('Success')
		else:
			print(response.status_code, response.reason)	

	@staticmethod
	def get_past_locations(platform_id):
		class PredictionList(list):

			def return_lats(self):
				return [x['lat'] for x in self]

			def return_lons(self):
				return [x['lon'] for x in self]

			def return_time(self):
				date_string_list = [x['datetime'] for x in self]
				return [datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S+00:00') for x in date_string_list]

			def return_depth(self):
				return [x['park_pressure'] for x in self]


		response = requests.get(SiteAPI.api_url + 'meta/' + platform_id,
								params={'limit': 50})
		if response.status_code == 200:
			holder = []
			data = response.json()
			for x in [x for x in data if x['datetime']!=None]:
				try: 
					x['park_pressure']
					holder.append(x)
				except KeyError:
					continue
			return PredictionList(holder[::-1])

		else:
			print(response.status_code, response.reason)

	@staticmethod
	def position_predictions(platform_id):
		response = requests.get(SiteAPI.api_url + 'position_predictions/' + platform_id)
		if response.status_code == 200:
			data = response.json()
			for r in data:
				print(r)
		else:
			print(response.status_code, response.reason)

	@staticmethod
	def delete_by_model(model,platform_id):
		model_dict = {'model': model}
		response = requests.delete(SiteAPI.api_url + 'position_predictions/' + platform_id,
								headers={'Authorization': SiteAPI.access_token},
								params=model_dict)
		if response.status_code == 200:
			print('Success')
		else:
			print(response.status_code, response.reason)
