import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# histograms of hits in hodoscope from experiment in CERN
data = pd.read_parquet('data_BM01P1_hits.parquet')      

# DATA PREP

# coding names for histograms (could be useful for further development)
def make_names(len_names):
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # len = 26
    names = []
    j = 0

    for i in range(len_names):
        names.append(abc[j % len(abc)] + abc[i % len(abc)])
        if i % 26 == 0 and i != 0:
            j += 1
            if j > 25:
                print('No more names left.')
                break
    return names

def modify_dataset(df, names):
    
    dataset_tmp = {}
    for col in df.columns:
        dataset_tmp[names[col]] = df[col].values

    df = pd.DataFrame(dataset_tmp)
    return df

def normalize(vec):
    top = max(vec)
    for i in range(len(vec)):
        vec[i] = vec[i]/top
    return vec

dataset = {}

for i in range(len(data.values)):
    dataset[i] = normalize(data.T[i])

norm_data = pd.DataFrame(data=dataset)

names = make_names(len(data))
norm_data = modify_dataset(norm_data, names)    
norm_data = norm_data.T
# END OF DATA PREP

# helping functions (could've been programmed as lambdas)
def dist(ptA, ptB):
    return np.linalg.norm(ptA-ptB)

def avg(container):
    return sum(container)/len(container)

# function calculating average distances of each point to his closest k neighbors
# used below with k = len(data) 
def calculate_avg_distances(test_dataset, k):

    temp_distances = []
    average_distances = []

    for ptA in test_dataset.values:
        for ptB in test_dataset.values:
            if not ptA is ptB:
                temp_distances.append(dist(ptA, ptB))
        temp_distances.sort()
        average_distances.append(avg(temp_distances[:k]))
        temp_distances.clear()
    return average_distances

# function used for debugging (not important)
def first_approach(data, k):
    avg_test_distances = calculate_avg_distances(data, k)
    mean_length = avg(avg_test_distances)
    outlier_cnt = 0
    return mean_length, avg_test_distances, outlier_cnt

if __name__ == '__main__':
    
    avg_distances = calculate_avg_distances(norm_data, 64)
    mean = avg(avg_distances)

    # list of measure of outlierness
    deviations = [abs(d-mean) for d in avg_distances]       
    
    idxs = [i for i in range(len(deviations))]
    
    # plotting the measure of outlierness of each histogram 
    plt.scatter(idxs, deviations)
    plt.xlabel('Histogram index')
    plt.ylabel('Measure of outlierness')
    plt.title('Distances of histograms from avg deviation')
    plt.show()
    # data concentrates around 0.08 leaving out 2 data points (on index 2 and 6) which may be classified as outliers
    
    n_outliers = sum([1 for i in deviations if i>=0.2])
    print(f'Found {n_outliers} outliers.')
    
    # cycle iterating over decision boundary values to plot graph of discovered outliers to each k 
    # (for visual fine tuning of k paramenter)
    for dec_boundary in range(0,5):
        outs = []
        dec_boundary /= 10
        for k in range(1,len(data)):
            avg_distances = calculate_avg_distances(norm_data, k)
            mean = avg(avg_distances)
            deviations = [abs(d-mean) for d in avg_distances]       
            idxs = [i for i in range(len(deviations))]
            outs.append(sum([1 for i in deviations if i>=dec_boundary]))
        plt.plot(outs)
        plt.title(f'Number of outliers against k for dec boundary {dec_boundary}')
        plt.xlabel('k')
        plt.ylabel('# outliers')
        plt.show()