import numpy as np

class spreadingFunction:

    def __init__(self):
        self.alpha = 0.8
        self.bark_resolution = 48
        pass


    def spreadVector(self, barkResolution, frequency):

        spreading_vector =np.zeros(2* self.bark_resolution)
        sim_masking=23.5
        spreading_vector[0:self.bark_resolution]=np.linspace(-frequency*27,-8,self.bark_resolution)-sim_masking
        spreading_vector[self.bark_resolution: 2*self.bark_resolution] = np.linspace(0,-frequency*12,self.bark_resolution)\
                                                                         -sim_masking

        return spreading_vector

    def spreadingMatrix(self, spreading_vector,barkResolution):

        spread_volage_vector = 10.0 ** (spreading_vector/20.0** self.alpha)
        spreading_matrix = np.zeros(barkResolution,barkResolution)
        for i in range(barkResolution):
            spreading_matrix[i,:] = spread_volage_vector[(barkResolution-i):(2*barkResolution-i)]

        return spreading_matrix
