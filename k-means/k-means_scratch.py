import numpy as np
import matplotlib.pyplot as plt
from random import randint, seed

# mean object class, holding seld coordinates and current nearest points to it
class Mean():
    def __init__(self):
        self.mean = np.array([randint(40,90),randint(80,180)])
        self.nearest_pts = []
    
    def add_closest_point(self, pt):
        self.nearest_pts.append(pt)
    
    def calc_new_mean(self):
        self.mean = np.array(sum(self.nearest_pts)/len(self.nearest_pts))
    
    def __str__(self):
        return f'Mean at coords: {self.mean}'

def dist(ptA : np.array, ptB : np.array) -> float:
    return np.linalg.norm(ptA-ptB)

def find_nearest_cluster(pt, means):
    distances = [dist(pt,m.mean) for m in means]
    return means[distances.index(min(distances))]

def visually_classify(k, means, epochs):
    epoch = 0
    while epoch < epochs:
        plt.scatter(apples_train[:, 0], apples_train[:, 1], color='red', label='Apples')
        plt.scatter(peaches_train[:, 0], peaches_train[:, 1], color='orange', label='Peaches')
        plt.scatter(pears_train[:, 0], pears_train[:, 1], color='green', label='Pears')

        for pt in data_train:
            mean = find_nearest_cluster(pt, means)
            mean.add_closest_point(pt)

        for m in means:
            plt.scatter(m.mean[0], m.mean[1], color='black', marker='x')
            m.calc_new_mean()

        plt.title('Data')
        plt.grid(True)
        plt.pause(0.01)
        plt.clf()
        epoch += 1


if __name__ == '__main__':
         
    np.random.seed(42)
    seed(42)

    # generating toy dataset (toy dataset was used in kNN) in this implementation we do not require 'labels'
    apples = np.random.normal(loc=[120, 160], scale=[13, 15], size=(60, 2))
    peaches = np.random.normal(loc=[90, 115], scale=[7, 13], size=(60, 2))
    pears = np.random.normal(loc=[70, 90], scale=[15, 18], size=(60, 2))
    
    apples_train = apples[:40]
    peaches_train = peaches[:40]
    pears_train = pears[:40]
    
    # test data
    apples_test = apples[40:]
    peaches_test = peaches[40:]
    pears_test = pears[40:]

    # datasets: all three classes merged together - essential as an input for get_neighors()
    data_train = np.vstack((apples_train, peaches_train, pears_train))
    k = 3
    
    # creating means 
    means = [Mean() for _ in range(k) ]
    
    # run classification with visual output
    visually_classify(k=k, means=means, epochs=50)
    
    plt.scatter(apples_train[:, 0], apples_train[:, 1], color='red', label='Apples')
    plt.scatter(peaches_train[:, 0], peaches_train[:, 1], color='orange', label='Peaches')
    plt.scatter(pears_train[:, 0], pears_train[:, 1], color='green', label='Pears')
    
    for m in means:
        plt.scatter(m.mean[0], m.mean[1], color='black', marker='x')
        m.calc_new_mean()

    #plt.title('Final Data')
    #plt.grid(True)
    plt.show()
    
    