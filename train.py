from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP

def split_data(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size = 10000, random_state=123, stratify=y )
    
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 5000, random_state=123, stratify=y_temp)
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():



    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X = X.values
    y = y.astype(int).values

    print(X.shape)
    print(y.shape)


    X = ((X/255.) - 0.5)*2
    ''' Visualize dataset 
    '''
    # fig, ax = plt.subplots(nrows=5, ncols=5, sharex = True, sharey = True)
    # ax = ax.flatten()
    # for i in range(25):
    #     img = X[y==7][i].reshape(28,28)
    #     ax[i].imshow(img, cmap = 'Greys')
        
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()
    X_train, y_train, X_val, y_val, X_test, y_test  = split_data(X,y)
    
    
    
if __name__ == "__main__":
    main()