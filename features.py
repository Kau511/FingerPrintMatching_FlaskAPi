import helper
import numpy as np




def get_features(fincoords, vector, orientations):
	"""
	Returns a list of objects of Feature Bases
	"""

	features = []
	feature_obj = []
	for i,point in enumerate(fincoords):
		feature = FeatureBase(np.array([point[0], point[1]]), point[2], orientations)
		vec = feature.make_feature_vector(vector[i])
		features.append(vec)
		feature_obj.append(feature)

	return features, feature_obj

class FeatureBase:
	"""
	Represents a feature as Fk = (xk,yk,fik,tk)
	"""
	def __init__(self, point, type, orientations):
		self.point =  point
		self.orientations = orientations
		self.theta = self.orientations[point[0].astype(int), point[1].astype(int)]
		self.type = int(type==1)

	def make_feature_vector(self, ridgecount):

		ki = ridgecount[0]
		kj = ridgecount[3]

		itheta, jtheta = self.orientations[ki[0].astype(int),ki[1].astype(int)], self.orientations[kj[0].astype(int),kj[1].astype(int)]

		self.dki = helper.dki(self.point,ki)
		self.dkj = helper.dki(self.point,kj)

		diff_ki = (self.point - ki)
		diff_kj = (self.point - kj)

		self.fiki = helper.dfi(np.arctan2(diff_ki[1],diff_ki[0]),self.theta) 
		self.fikj = helper.dfi(np.arctan2(diff_kj[1],diff_kj[0]),self.theta)

		self.phiki = helper.dfi(itheta, self.theta)
		self.phikj = helper.dfi(jtheta, self.theta)

		self.nki = ridgecount[2]
		self.nkj = ridgecount[5]

		self.typei = int(ridgecount[1]==1)
		self.typej = int(ridgecount[4]==1)

		return (self.dki,
				self.dkj,
				self.fiki,
				self.fikj,
				self.phiki,
				self.phikj,
				self.nki,
				self.nkj,
				self.type,
				self.typei,
				self.typej
			)


class FeaturePolar:

	def __init__(self,current,base):

		self.current = current
		self.base = base

	def convert(self):

		pb = self.base.point
		pk = self.current.point
		diff_kb = pk-pb
		self.rkb = helper.dki(pb,pk)
		self.fikb = helper.dfi(np.arctan2(diff_kb[1],diff_kb[0]),self.base.theta)
		self.phikb = helper.dfi(self.current.theta,self.base.theta)

		return np.array((self.rkb,
				self.fikb,
				self.phikb
			))


 

