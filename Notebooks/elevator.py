import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
 

exp_factor = 2.0

def houroftheday(x):
	tmp = datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
	return str(tmp.hour)

def date(x):
	tmp = datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
	return str(tmp.date())
	       



if __name__ == '__main__':

	fdir = '/Users/naveenkumar/Dropbox/OnlineWork/Datasets/ElevatorProblem/'
	ds = pd.read_csv('input.csv')
 


	FloorInAccess = ds[['Person_Id', 'Floor_in', 'Elevator_in_time']]
	FloorOutAccess = ds[['Person_Id', 'Floor_out', 'Elevator_in_time']]
    
	FloorInAccess.rename(columns={'Floor_in':'Floor_id'}, inplace=True)
	FloorOutAccess.rename(columns={'Floor_out':'Floor_id'}, inplace=True)
     
	AllFloorAccessed = pd.concat([FloorInAccess, FloorOutAccess])
	AllFloorAccessed['Elevator_in_date'] = AllFloorAccessed['Elevator_in_time'].apply(lambda  x:(date(x)))
	print AllFloorAccessed
	 
	UniqueFloorAccessed = AllFloorAccessed.drop_duplicates(['Person_Id', 'Floor_id', 'Elevator_in_date'])
	UniqueUsersPerFloor = UniqueFloorAccessed.groupby(['Floor_id']).Person_Id.nunique().reset_index()
	UniqueUsersPerFloor.rename(columns={'Person_Id' : 'UniquePersons'}, inplace = True)

	FloorAccesses = UniqueFloorAccessed.groupby(['Person_Id', 'Elevator_in_date']).Floor_id.nunique().reset_index()
	FloorAccesses.rename(columns={'Floor_id' : 'DayCountOfFloorAccesses'}, inplace = True)
	PersonAvgFloorAccesses = FloorAccesses.groupby(['Person_Id'])['DayCountOfFloorAccesses'].mean().reset_index()
	PersonAvgFloorAccesses.rename(columns={'DayCountOfFloorAccesses' : 'AvgNumFloorAccesses'}, inplace = True)
	print FloorAccesses
	print PersonAvgFloorAccesses

	UniqueFloorAccessed_count = UniqueFloorAccessed[['Person_Id', 'Floor_id']].groupby(['Person_Id'], as_index=False).count()
	UniqueFloorAccessed_count.rename(columns={'Floor_id':'NumUniqueFloorsAccessed'}, inplace=True)





	#Oddness scores...
	EleAccessHours = ds[['Person_Id', 'Elevator_in_time']]
	EleAccessHours['Ele_Access_Hour'] = EleAccessHours['Elevator_in_time'].apply(lambda col:houroftheday(col))
	EleAccessHours['Ele_Access_Date'] = EleAccessHours['Elevator_in_time'].apply(lambda col:date(col))
	print(EleAccessHours)
	#Hour Oddness Score

	print '-' * 100
	EleAccessHours_count = EleAccessHours[['Person_Id', 'Ele_Access_Hour']].groupby(['Ele_Access_Hour'], as_index=False).count()
	EleAccessHours_count.rename(columns={'Person_Id':'NumTimeFloorAccessed'}, inplace=True)
	EleAccessHours_count['HourOddnessScore'] = np.power(EleAccessHours_count['NumTimeFloorAccessed'].rank(ascending = False), exp_factor)


	EleAccessHoursOddness = pd.merge(EleAccessHours, EleAccessHours_count[['Ele_Access_Hour', 'HourOddnessScore']], on='Ele_Access_Hour', how='left')
	print(EleAccessHoursOddness)
	HourOddnessScore = EleAccessHoursOddness[['Person_Id', 'HourOddnessScore']].groupby(['Person_Id'], as_index=False).mean()
	print(HourOddnessScore)

	print '-' * 100
	#HourFloor Oddness...
	EleAccessHours = ds[['Person_Id', 'Elevator_in_time', 'Floor_in']]
	EleAccessHours['Ele_Access_Hour'] = EleAccessHours['Elevator_in_time'].apply(lambda col:houroftheday(col))

	EleAccessHours_count = EleAccessHours[['Person_Id', 'Floor_in', 'Ele_Access_Hour']].groupby(['Ele_Access_Hour', 'Floor_in'], as_index=False).count()
	EleAccessHours_count.rename(columns={'Person_Id':'NumTimeFloorAccessedInHour'}, inplace=True)
	EleAccessHours_count['FloorHourOddnessScore'] = np.power(EleAccessHours_count['NumTimeFloorAccessedInHour'].rank(ascending = False), exp_factor)


	EleAccessHoursOddness = pd.merge(EleAccessHours, EleAccessHours_count[['Ele_Access_Hour', 'FloorHourOddnessScore']], on='Ele_Access_Hour', how='left')
	print(EleAccessHoursOddness)
	HourFloorOddnessScore = EleAccessHoursOddness[['Person_Id', 'FloorHourOddnessScore']].groupby(['Person_Id'], as_index=False).mean()
	print(HourFloorOddnessScore)
	print '-' * 100



	FeatureSpace_ = PersonAvgFloorAccesses.merge(HourOddnessScore, left_on = 'Person_Id', right_on = 'Person_Id', how = 'left')

	FeatureSpace  = FeatureSpace_.merge(HourFloorOddnessScore, left_on = 'Person_Id', right_on = 'Person_Id', how = 'left')
	print FeatureSpace
	print FeatureSpace.shape


	X = FeatureSpace[['HourOddnessScore', 'FloorHourOddnessScore']].values

	import numpy as np
	from sklearn.decomposition import PCA

	pca = PCA(n_components=1)
	pca.fit(X)

	print(pca.explained_variance_ratio_)  
	
	ReducedFeature = pd.DataFrame(pca.transform(X), columns = ['PCA_reduced_OddnessFeature'])

	FeatureSpace = pd.concat([FeatureSpace, ReducedFeature], axis = 1)
	print FeatureSpace

	from sklearn.cluster import AgglomerativeClustering
	C = AgglomerativeClustering()
	ML_in = FeatureSpace[['AvgNumFloorAccesses', 'HourOddnessScore', 'FloorHourOddnessScore', 'PCA_reduced_OddnessFeature']].values

	result = C.fit_predict(ML_in)

	print result
	print dir(result)
	print result.tolist()
	print dir(C)

	print C.compute_full_tree




