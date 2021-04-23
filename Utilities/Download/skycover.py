import xml.etree.ElementTree as ET
from urllib.request import urlopen
from decimal import Decimal


def get_address(lat,lon):
	TWOPLACES = Decimal(10)**-2
	address = 'https://preview.weather.gov/xml/SOAP_server/ndfdXMLclient.php?whichClient=NDFDgenLatLonList&lat=&lon=&listLatLon=' \
	+str(Decimal(lat).quantize(TWOPLACES)) \
	+'%2C' \
	+str(Decimal(lon).quantize(TWOPLACES)) \
	+'&lat1=&lon1=&lat2=&lon2=&resolutionSub=&listLat1=&listLon1=&listLat2=&listLon2=&resolutionList=&endPoint1Lat=&endPoint1Lon=&endPoint2Lat=&endPoint2Lon=&listEndPoint1Lat=&listEndPoint1Lon=&listEndPoint2Lat=&listEndPoint2Lon=&zipCodeList=&listZipCodeList=&centerPointLat=&centerPointLon=&distanceLat=&distanceLon=&resolutionSquare=&listCenterPointLat=&listCenterPointLon=&listDistanceLat=&listDistanceLon=&listResolutionSquare=&citiesLevel=&listCitiesLevel=&sector=&gmlListLatLon=&featureType=&requestedTime=&startTime=&endTime=&compType=&propertyName=&product=time-series&begin=2004-01-01T00%3A00%3A00&end=2025-02-26T00%3A00%3A00&Unit=e&sky=sky&Submit=Submit'
	return address

def get_skycover_profile(lat,lon)
	xml = urlopen(get_address(lat,lon))
	tree = ET.parse(xml)
	root = tree.getroot()