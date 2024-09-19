import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Question 1
#a)
std_dev = np.sqrt(2)
generator_0 = lambda n: np.array([np.array(np.random.normal(loc=(-2, 2), scale=(std_dev, std_dev))) for i in range(n)])
generator_1 = lambda n: np.array([np.array(np.random.normal(loc=(2, -2), scale=(std_dev, std_dev))) for i in range(n)])

#b)
sample_0 = generator_0(20)
sample_1 = generator_1(20)


# sns.scatterplot(x=sample_0[:,0], y=sample_0[:,1])
# sns.scatterplot(x=sample_1[:,0], y=sample_1[:,1])

#c)
# plt.show()

#question 2
#a)

set1 = np.array(
[[-0.04068806, -1.32374476],
 [ 1.07681611,0.39023218],
 [-0.26563335,0.4529352 ],
 [ 0.1399671,-1.47975074],
 [ 0.71763354, -1.30100248],
 [ 0.74041372 ,-1.3522152 ],
 [-0.02690943 ,-0.05805199],
 [ 0.65770564 ,-1.91897725],
 [ 1.68146628, -0.89579474],
 [ 0.32086282, -0.50824497],
 [-0.10373497, -2.77343626],
 [ 1.64918523, -0.50832955],
 [ 2.00334448,0.6657719 ],
 [ 1.70677317, -0.33095003],
 [ 2.13369501,0.01773743],
 [ 1.77972744, -1.58538463],
 [ 0.67738148, -1.52433779],
 [ 0.9492246,-1.13840792],
 [ 0.39413062,0.9738537 ]])
 
set2 = np.array([[
    -1.13399371,0.96717537],
 [-0.59367734,0.58606496],
 [-1.05583982 ,-0.13440433],
 [-1.63895513,1.69139435],
 [-1.44480522,0.7755843 ],
 [-0.07053615,0.83433966],
 [-2.58129507,0.55809511],
 [-1.76966915,0.75690371],
 [-1.15355414,0.02040755],
 [-1.2560966, 1.27145506],
 [ 0.30261796,0.80192355],
 [-1.1932258, 0.58107035],
 [ 0.3118735, 1.13077845],
 [-1.74286621,1.13429882],
 [-0.4829813, 1.78523918],
 [-1.90070164, -1.40386622],
 [ 0.57248805,1.88590498],
 [-0.27272491,1.86843519],
 [-1.2668738, 0.19743395],
 [-0.30808727,2.3399052 ]])

theta_0 = np.random.uniform(low=-0.5, high=0.5)
theta_1 = np.random.uniform(low=-0.5, high=0.5)
theta_2 = np.random.uniform(low=-0.5, high=0.5)
print(theta_0, theta_1, theta_2)


sns.scatterplot(x=set1[:,0], y=set1[:,1])
sns.scatterplot(x=set2[:,0], y=set2[:,1])
f = lambda X: theta_0 + theta_1*X[0] + theta_2*X[1]

X = [i for i in range(-3,4)]
Y = [f([i, i]) for i in range(len(X))]
# sns.lineplot(x=X, y=Y)

# plt.show()

theta_0 = -0.05991429899056311
theta_1 = -0.21581452483645724
theta_2 = -0.32796156736396775
X = [i for i in range(-3,4)]
Y = [f([i, i]) for i in range(len(X))]
f = lambda X: theta_0 + theta_1*X[0] + theta_2*X[1]

sigma = lambda X: 1/(1 + np.exp(theta_0 + theta_1*X[0] + theta_2*X[1]))
error_f = lambda x, y: y*np.log(sigma(x)) + (1-y)*np.log(1 - sigma(x))


#b) calculate error
log_likelihood = lambda X, Y: np.sum([error_f(X[i], Y[i]) for i in range(len(X))])

_y1, _y2 = np.zeros((len(set1))), np.ones((len(set2)))
_Y = np.concatenate((_y1, _y2))
_X = np.concatenate((set1, set2))
print(len(_X), len(_Y))
print(log_likelihood(_X, _Y))


def train_weights(theta0, theta1, theta2, X, Y):
    theta0 ,theta1, theta2
    sigma = lambda X: 1/(1 + np.exp(theta0 + theta1*X[0] + theta2*X[1]))
    old_thetas = np.array((theta0, theta1, theta2))
    tol = 115e-3
    a = 0.1
    it = 0
    max_iter = 1000
    while True:
        it +=1
        for i in range(len(X)):
        #theta 0
            theta0 = theta0 - a*(Y[i] - sigma(X[i]))
        #theta 1
            theta1 = theta1 - a*((Y[i] - sigma(X[i]))*X[i][0])
            
        #theta 2
            theta2 = theta2 - a*((Y[i] - sigma(X[i]))*X[i][1])
        
        print('epoch', it)
        sigma = lambda X: 1/(1 + np.exp(theta0 + theta1*X[0] + theta2*X[1]))
        error_f = lambda x, y: y*np.log(sigma(x)) + (1-y)*np.log(1 - sigma(x))

        #b) calculate normed diff
        new_thetas = np.array((theta0, theta1, theta2))
        diff = np.linalg.norm(old_thetas-new_thetas)
        if diff < tol or it > max_iter:
            print(theta0, theta1, theta2)
            return theta0, theta1, theta2 
  
        old_thetas = new_thetas 
    
        
Y = np.zeros((len(set1)))
theta_0, theta_1, theta_2 = train_weights(theta_0, theta_1, theta_2, set1, Y)

Y = np.ones((len(set2)))
theta_0, theta_1, theta_2 = train_weights(theta_0, theta_1, theta_2, set2, Y)



sns.scatterplot(x=set1[:,0], y=set1[:,1])
sns.scatterplot(x=set2[:,0], y=set2[:,1])
f = lambda X: theta_0 + theta_1*X[0] + theta_2*X[1]

X = [i for i in range(-3,4)]
Y = [f([i, i]) for i in range(len(X))]
sns.lineplot(x=X, y=Y)

# plt.show()
# sigma = lambda X: 1/(1 + np.exp(theta_0 + theta_1*X[0] + theta_2*X[1]))
# error_f = lambda x, y: y*np.log(sigma(x)) + (1-y)*np.log(1 - sigma(x))
# log_likelihood = lambda X, Y: np.sum([error_f(X[i], Y[i]) for i in range(len(X))])
# _y1, _y2 = np.zeros((len(set1))), np.ones((len(set2)))
# _Y = np.concatenate((_y1, _y2))
# _X = np.concatenate((set1, set2))
# print(len(_X), len(_Y))
# print(log_likelihood(_X, _Y))

#g)
set2 = np.array([[-2.94791504,  1.34764071],
 [-2.25271723,  2.76803535],
 [-1.7412961 ,  2.19408415],
 [-2.98577396 , 2.69801567],
 [-2.74955818 , 0.67989169],
 [-3.55784562 , 3.08848134],
 [-2.03088022 , 4.50084035],
 [-3.09556721 , 3.61807031],
 [ 0.04306024 , 5.76306771],
 [-2.70697528 , 2.68793311],
 [ 1.26464153 , 3.13317281],
 [-2.08127813 , 0.79064381],
 [ 2.41987296, -0.00632301],
 [-1.52413615 , 0.64233479],
 [-1.21743829 , 2.02410434],
 [-3.06704359  ,1.67138678],
 [-2.86204143 , 3.56010049],
 [-3.25347178,  1.60247596],
 [-1.00278941  ,1.34024802],
 [-4.43229393 , 0.32174078]])

set1 = np.array([[ 2.8719416,  -1.02811509],
 [ 0.53788966, -3.5069448 ],
 [ 5.42188577, -1.77723995],
 [ 1.70875584, -1.67793809],
 [ 2.27113537, -1.98848426],
 [ 4.51495345, -1.5931088 ],
 [ 0.29983661, -0.1607572 ],
 [ 1.35223362, -1.35524189],
 [ 0.31824194, -3.59040842],
 [ 0.7306305 , -3.9686611 ],
 [ 2.63920442 ,-3.68144712],
 [ 2.72069119, -0.18183426],
 [-0.26973053, -2.32053322],
 [ 2.66697819, -1.00348855],
 [ 2.85575122, -4.13159506],
 [ 0.11392122, -1.83161983],
 [ 3.11337754, -0.69668314],
 [ 3.8379036 , -1.72122869],
 [ 1.94352222, -1.21772446],
 [ 3.18014868 ,-2.12371249]])


# f = lambda X: theta_0 + theta_1*X[0] + theta_2*X[1]

# sigma = lambda X: 1/(1 + np.exp(theta_0 + theta_1*X[0] + theta_2*X[1]))
# error_f = lambda x, y: y*np.log(sigma(x)) + (1-y)*np.log(1 - sigma(x))

# log_likelihood = lambda X, Y: np.sum([error_f(X[i], Y[i]) for i in range(len(X))])

# _y1, _y2 = np.zeros((len(set1))), np.ones((len(set2)))
# _Y = np.concatenate((_y1, _y2))
# _X = np.concatenate((set1, set2))
# print(len(_X), len(_Y))
# print(log_likelihood(_X, _Y))

# predicted_labels = [np.round(sigma(_X[i])) for i in range(len(_X))]
# conf_matrix = confusion_matrix(_Y, predicted_labels)
# accuracy = np.trace(conf_matrix) /np.sum(conf_matrix)

# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(2), yticklabels=np.arange(2))
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# print('accuracy:', accuracy)
# plt.show()

set1 = generator_1(20)
set2 = generator_0(20)

print(set1, set2)


f = lambda X: theta_0 + theta_1*X[0] + theta_2*X[1]

sigma = lambda X: 1/(1 + np.exp(theta_0 + theta_1*X[0] + theta_2*X[1]))
error_f = lambda x, y: y*np.log(sigma(x)) + (1-y)*np.log(1 - sigma(x))

log_likelihood = lambda X, Y: np.sum([error_f(X[i], Y[i]) for i in range(len(X))])

_y1, _y2 = np.zeros((len(set1))), np.ones((len(set2)))
_Y = np.concatenate((_y1, _y2))
_X = np.concatenate((set1, set2))
print(len(_X), len(_Y))
print(log_likelihood(_X, _Y))

predicted_labels = [np.round(sigma(_X[i])) for i in range(len(_X))]
conf_matrix = confusion_matrix(_Y, predicted_labels)
accuracy = np.trace(conf_matrix) /np.sum(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(2), yticklabels=np.arange(2))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
print('accuracy:', accuracy)
plt.show()