import numpy as np
import matplotlib.pyplot as plt

# eucleid distatnce (could me altered to any desired metric)
def dist(ptA : np.array, ptB : np.array) -> float:
    return np.linalg.norm(ptA-ptB)

# alternative approach to finding nearest points
def get_neighbors_alternative(new_point : np.array, pts : np.array, k : int) -> list:
    temp_dict = {}
    neigh = []
    for i in range(len(pts[:-1])):
        temp_dict[i] = dist(pts[i], new_point)
        if len(neigh) < k:
            neigh.append(pts[i])
        elif max(max_list:=[dist(neigh[k],new_point) for k in range((len(neigh)))]) > dist(pts[i], new_point):
            idx = max_list.index(max(max_list))
            neigh[idx] = pts[i]
    return neigh

# function that finds nearest points (efficient implementation)
def get_neighbors(new_point : np.array, pts : np.array, k : int) -> list:
    neigh = sorted(pts[:-1], key=lambda x: dist(x,new_point))
    return neigh[:k]

# function that classifies neighbors 
# (implemented this way due to charaacter of the toy dataset)
def classify(fruit : str, neigh : list) -> bool:
    classes = {'apple' : 0, 'peach' : 0, 'pear' : 0}
    for n in neigh:
        if n in apples_train:
            classes['apple'] += 1 
        elif n in peaches_train:
            classes['peach'] += 1
        elif n in pears_train:
            classes['pear'] += 1
        
    if max(classes, key=classes.get) == fruit:
        global correct
        correct += 1
        return True
    else:
        global incorrect
        incorrect += 1
        return False

# given param k classifies each point 
# (as data is split into 3 datasets algorithm iterates gradually through each dataset - again due character of the toy dataset)
def classify_test_data(k=3) -> None:
    for i in range(len(apples_test)):
        neigh = get_neighbors(apples_test[i], data_train, k)
        classify('apple', neigh)
    
    for i in range(len(pears_test)):
        neigh = get_neighbors(pears_test[i], data_train, k)
        classify('pear', neigh)    

    for i in range(len(peaches_test)):
        neigh = get_neighbors(peaches_test[i], data_train, k)
        classify('peach', neigh)
        
    #print(f'Correct: {correct} | Incorrect: {incorrect}')

if __name__ == '__main__':

    # toy dataset inicialization below
    np.random.seed(42)

    # generated data split into classes
    apples = np.random.normal(loc=[75, 150], scale=[10, 20], size=(60, 2))
    peaches = np.random.normal(loc=[60, 135], scale=[8, 15], size=(60, 2))
    pears = np.random.normal(loc=[70, 120], scale=[9, 18], size=(60, 2))
    
    # training splits (66 to 33 split was chosen)
    apples_train = apples[:40]
    peaches_train = peaches[:40]
    pears_train = pears[:40]

    # test data
    apples_test = apples[40:]
    peaches_test = peaches[40:]
    pears_test = pears[40:]

    # datasets: all three classes merged together - essential as an input for get_neighors()
    data_train = np.vstack((apples_train, peaches_train, pears_train))

    plt.scatter(apples_train[:, 0], apples_train[:, 1], color='red', label='Apples')
    plt.scatter(peaches_train[:, 0], peaches_train[:, 1], color='orange', label='Peaches')
    plt.scatter(pears_train[:, 0], pears_train[:, 1], color='green', label='Pears')
    
    plt.title('Data')
    plt.grid(True)
    plt.show()
    
    # classification counters
    correct = 0
    incorrect = 0

    i = 1
    performance = []
    while i <= 19:
        classify_test_data(k=i)
        # performence is measured as percentage of incorrect guesses
        performance.append(1-correct/(correct+incorrect))
        correct = 0
        incorrect = 0
        i+=2
    
    # due to character of performence measure we seek minimum (of incorrect guesses)
    best_performing = min(performance)
    # console output
    print(f'Best performing at k={2*performance.index(best_performing)+1} with \n\
          correct guesses: {3*len(apples_test) - round(3*len(apples_test)*best_performing)}\n\
          incorrect guesses: {round(3*len(apples_test)*best_performing)}')
    
    plt.style.use('Solarize_Light2')
    plt.title('Performance')
    plt.plot([2*i+1 for i in range(len(performance))],performance)
    plt.show()
