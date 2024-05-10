# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.spatial.distance import cdist

import itertools

# from haversine import haversine, Unit

def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))

def composed_haversine_distance(x, xi, radius):
    lat1, lon1, lat2, lon2 = x[0],x[1],xi[0],xi[1]
    
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
    
    a = dLat * radius
    b = dLon * radius
    
    return a, b

def haversine_distance(x, xi, radius):
    lat1 = x[0]
    lon1 = x[1]
    lat2 = xi[0]
    lon2 = xi[1]
    
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
 
    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
 
    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
             math.cos(lat1) * math.cos(lat2));
    rad = radius
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


def gaussian_kernel_2D(distanceLat, distanceLon, bandwidthLat, bandwidthLon, area=None):
    
    frac_x = (distanceLon / bandwidthLon)**2
    frac_y = (distanceLat / bandwidthLat)**2
    
    exponential = np.exp(-0.5*(frac_x + frac_y))
    
    
    Amplitude = 1
    if area is not None:
        Amplitude = area
    #     Amplitude = 1 # Turn this into a function of the Area?
    
    val = Amplitude * exponential
    return val

def gaussian_kernel(distance, bandwidth):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val


# +
class Mean_Shift:
    def __init__(self, look_distance, kernel_bandwidthLon, kernel_bandwidthLat, radius, n_iterations=5, angle=0, max_scaled_area_muHem=2000):
        self.history = []
        self.n_iterations = n_iterations
        self.look_distance = look_distance  # How far to look for neighbours.
        self.kernel_bandwidthLon = kernel_bandwidthLon  # Longitude Kernel parameter.
        self.kernel_bandwidthLat = kernel_bandwidthLat  # Latitude Kernel parameter.
        self.radius = radius
        self.centroids = []
        self.merge_dist = look_distance
        self.angle = angle
        self.max_scaled_area_muHem = max_scaled_area_muHem


    def find_neighbours(self, data, areas, x_centroid):

        cos_angle = np.cos(np.radians(180.-self.angle))
        sin_angle = np.sin(np.radians(180.-self.angle))

        g_ell_width = 2. * self.kernel_bandwidthLon
        g_ell_height = 2. * self.kernel_bandwidthLat

        xc = data[:,0] - x_centroid[0]
        yc = data[:,1] - x_centroid[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle 

        rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)

        neighbours = self.data[np.where(rad_cc <= 1.0)[0]]

        neighbours_areas = np.array(areas)[rad_cc <= 1.0]

        return neighbours, neighbours_areas

    def get_area_weighted_ellipsis_width(self, current_area, all_areas):
        # area must be given in mu_hem and NOT IN PIXELS
        
        proportional_width = None
        
        if self.kernel_bandwidthLon >= self.kernel_bandwidthLat:
            lon_max = self.kernel_bandwidthLon
            lon_min = self.kernel_bandwidthLat + (lon_max - self.kernel_bandwidthLat)/4.
            
            if current_area <= self.max_scaled_area_muHem:
                rate = (current_area )/(self.max_scaled_area_muHem + 1e-10)
                proportional_width = lon_min + rate * (lon_max - lon_min)
    
            else:
                proportional_width = self.kernel_bandwidthLon
            
            assert (proportional_width >= lon_min) and (proportional_width <= lon_max)
        else:
            
            proportional_width = self.kernel_bandwidthLon
            
        assert proportional_width is not None
            
        return proportional_width

    #V2: weight ellipses width (and height?) by sunspot area
    def find_neighbours2(self, data, areas, x_centroid, x_centroid_area):
        # x_centroid is of of the form [lat, lon]

        # print(data)
        cos_angle = np.cos(np.radians(180.-self.angle))
        sin_angle = np.sin(np.radians(180.-self.angle))

        a = self.get_area_weighted_ellipsis_width( x_centroid_area, self.areas)
        b = self.kernel_bandwidthLat

        yc = data[:,0] - x_centroid[0] # yc = position of the sunspot in the latitude axis of the ellipse
        xc = data[:,1] - x_centroid[1] # xc = position of the sunspot in the Longitude axis of the ellipse

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle 

        rad_cc = (xct**2/(a)**2) + (yct**2/(b)**2)

        neighbours = self.data[rad_cc <= 1.0]

        neighbours_areas = np.array(areas)[rad_cc <= 1.0]

        return neighbours, neighbours_areas
            

    def fit(self,X, areas = None):
        # print(areas)
        if areas is not None:
            assert len(X) == len(areas)

        self.centroids = X.copy()
        self.data = X.copy()
        self.areas = areas.copy()

        # check if data in radians at wrap-around
        recentered = False
        if np.max(self.data[:,1]) - np.min(self.data[:,1]) > np.pi:
            # recenter data
            self.data[:,1] = (self.data[:,1] + np.pi) % (2*np.pi) 
            self.centroids[:,1] = (self.centroids[:,1] + np.pi) % (2*np.pi) 
            recentered = True
            
        self.history.append(np.copy(self.data))
        # for it in tqdm(range(self.n_iterations)):
        for it in range(self.n_iterations):
            # print('Iteration: ', it)
            for i, x in enumerate(self.centroids):
                ### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
                # neighbours, neighbours_areas = self.find_neighbours(self.data, self.areas, x)
                neighbours, neighbours_areas = self.find_neighbours2(self.data, self.areas, x, self.areas[i])
                # print('Neighbours: ', neighbours)

                ### Step 2. For each datapoint x ∈ X, compute m(x) = ∑ w(x, y) y / ∑ w(x, y).
                # mean_x = np.mean(neighbours[:,0])
                # mean_y = np.mean(neighbours[:,1])
                # new_x = np.array([mean_x, mean_y])

                #V2: weighted mean
                mean_x = np.sum(neighbours_areas * neighbours[:,0]) / np.sum(neighbours_areas)
                mean_y = np.sum(neighbours_areas * neighbours[:,1]) / np.sum(neighbours_areas)
                new_x = np.array([mean_x, mean_y])
                    

                ### Step 3. For each datapoint x ∈ X, update x ← m(x).
                self.centroids[i] = new_x
            self.history.append(np.copy(self.centroids))
            
        ## filter out duplicates
            
        X_copy = self.centroids.copy()
        X_copy = np.unique(X_copy, axis=0)

        combinations = itertools.combinations(X_copy, 2) 
        
        to_remove = []
        for c1,c2 in combinations:
            # d = haversine_distance(c1, c2, sunspots_sk.radius.km[0])
            # d = haversine_distance(c1, c2, self.radius)
            d = euclid_distance(c1, c2)
            # print(c1,c2, '->', d)
            if (d < self.merge_dist) and (c1.tolist() not in to_remove) :
                to_remove.append(c1.tolist())
        # print(to_remove)
        
        for duplicates in to_remove:
            X_copy = np.delete(X_copy, np.where((X_copy == duplicates).all(axis=1)), axis=0)
            # X_copy = np.unique(X_copy, axis=0)
        
        if recentered:
            X_copy[:,1] = (X_copy[:,1] - np.pi) % (2*np.pi)

        self.centroids = np.array(X_copy)


        # filter out centroids that have no neighbours  
        pred_data = self.data.copy()  
        if recentered:
            pred_data[:,1] = (pred_data[:,1] - np.pi) % (2*np.pi)
        data_classif = self.predict(pred_data)
        orig_len = len(self.centroids)
        orig_centroids = self.centroids.copy()
        # print(data_classif, orig_len)
        for i in range(orig_len):
            # print(i)
            if i not in data_classif:
                self.centroids = np.delete(self.centroids, np.where((self.centroids == orig_centroids[i]).all(axis=1)), axis=0)
        


    def set_centroids(self, new_centroids):
        self.centroids = new_centroids

    def predict(self,data):

        recentered = False
        if np.max(data[:,1]) - np.min(data[:,1]) > np.pi :
            # recenter data
            data[:,1] = (data[:,1] + np.pi) % (2*np.pi)
            self.centroids[:,1] = (self.centroids[:,1] + np.pi) % (2*np.pi)
            recentered = True
      
        #compare distance to either centroid
        distances = np.array([
                        [
                            # haversine_distance(x, centroid, sunspots_sk.radius.km[0])
                            # haversine_distance(x, centroid, self.radius)
                            euclid_distance(x, centroid) 
                         for x in data]  
                    for centroid in self.centroids])
   
        classification = (np.argmin(distances, axis=0))

        if recentered:
            self.centroids[:,1] = (self.centroids[:,1] - np.pi) % (2*np.pi)


        return classification

