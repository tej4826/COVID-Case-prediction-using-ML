import numpy as np
import csv
import sys

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_pr.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights

def apply_one_hot_encoding(X):
    unique_values = list(set(X))
    unique_values.sort()
    one_hot_encoding_map = {}
    counter = 0
    for x in unique_values:
        one_hot_encoding_map[x] = [0 for i in range(len(unique_values))]
        one_hot_encoding_map[x][counter] = 1
        counter += 1
    one_hot_encoded_X = []
    for x in X:
        one_hot_encoded_X.append(one_hot_encoding_map[x])

    one_hot_encoded_X = np.array(one_hot_encoded_X, dtype=int)
    return one_hot_encoded_X

def convert_given_cols_to_one_hot(X, column_indices):
    one_hot_encoded_X = np.zeros([len(X),1])

    start_index = 0
    #acts column pointer in X

    for curr_index in column_indices:
        #adding the columns present before curr_index in X (and not present in one_hot_encoded_X), to one_hot_encoded_X
        one_hot_encoded_X=np.append(one_hot_encoded_X,X[:, start_index:curr_index], axis=1)
        
        #applying one hot encoding for current column
        one_hot_encoded_column = apply_one_hot_encoding(X[:,curr_index])

        #appending the obtained one hot encoded array to one_hot_encoded_X
        one_hot_encoded_X=np.append(one_hot_encoded_X,one_hot_encoded_column, axis=1)

        #moving the column pointer of X to next current_index
        start_index = curr_index+1

    #adding any remaining columns to one_hot_encoded_X    
    one_hot_encoded_X=np.append(one_hot_encoded_X,X[:,start_index:], axis=1)
    one_hot_encoded_X = one_hot_encoded_X[:,1:]
    return one_hot_encoded_X

def get_correlation_matrix(X):
    num_vars = len(X[0])
    m = len(X)
    correlation_matix = np.zeros((num_vars,num_vars))
    for i in range(0,num_vars):
        for j in range(i,num_vars):
            mean_i = np.mean(X[:,i])
            mean_j = np.mean(X[:,j])
            std_dev_i = np.std(X[:,i])
            std_dev_j = np.std(X[:,j])
            numerator = np.sum((X[:,i]-mean_i)*(X[:,j]-mean_j))
            denominator = (m)*(std_dev_i)*(std_dev_j)
            corr_i_j = numerator/denominator    
            correlation_matix[i][j] = corr_i_j
            correlation_matix[j][i] = corr_i_j
    return correlation_matix

def select_features(corr_mat, T1, T2):
    n=len(corr_mat)
    filtered_features = []
    for i in range(1,n):
        if (abs(corr_mat[i][0]) > T1):
            filtered_features.append(i)
    m = len(filtered_features)
    removed_features = []
    selected_features = list(filtered_features)
    for i in range(0,m):
        for j in range(i+1,m):
            f1 = filtered_features[i]
            f2 = filtered_features[j]
            if (f1 not in removed_features and f2 not in removed_features): 
                if (abs(corr_mat[f1][f2]) > T2):
                    selected_features.remove(f2)
                    removed_features.append(f2)

    return selected_features

def replace_null_values_with_mean(X,real_X):
    #Obtain mean of columns
    col_mean = np.nanmean(real_X, axis=0)

    #Find indicies that we need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])
    return X

def mean_normalize(X, column_indices):
    X[:,column_indices]=((X[:,column_indices]-np.mean(X,axis=0)[column_indices])/((np.max(X,axis=0)[column_indices]-np.min(X,axis=0)[column_indices])))
    return X

def predict_target_values(test_X, weights):
    k=[]
    s=[]
    print(weights[0][:])
    for i in range(2):
        b=weights[i][-1]
        Z=np.dot(test_X,weights[i][:-1])+b
        A=sigmoid(Z)
        k.append(A)
    k=np.array(k)
    size=(k.shape)[1]
    for i in range(size):
        m=max(k[0][i],k[1][i])
        if m==k[0][i]:
            s.append([0])
        else:
            s.append([1])
    s=np.array(s)
    return s

    """
    Note:
    The preprocessing techniques which are used on the train data, should also be applied on the test 
    1. The feature scaling technique used on the training data should be applied as it is (with same mean/standard deviation/min/max) on the test data as well.
    2. The one-hot encoding mapping applied on the train data should also be applied on test data during prediction.
    3. During training, you have to write any such values (mentioned in above points) to a file, so that they can be used for prediction.
     
    You can load the weights/parameters and the above mentioned preprocessing parameters, by reading them from a csv file which is present in the preprocessing_regularization.zip
    """
    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in preprocessing_regularization.zip and imported properly.
    """
def min_max_normalize(X, column_indices):
    for column_index in column_indices:
        column = X[:,column_index]
        min = np.min(column, axis=0) 
        max = np.max(column, axis=0)
        difference = max- min
        X[:,column_index] = (column - min) /difference
    return X

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    real_X,_= import_data_and_weights("train_X_pr.csv", "WEIGHTS_FILE.csv")
    X = test_X
    X = replace_null_values_with_mean(X,real_X)
    X = convert_given_cols_to_one_hot(X, [0,3])
    m=(X.shape)[0]
    n=(X.shape)[1]
    col=[]
    for i in range(n):
        for j in range(m):
            if X[j][i]>1 or X[j][i]<0:
                col.append(i)
                break
    X = min_max_normalize(X, col)
    #corr_mat = get_correlation_matrix(X)
    fea = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    count=0
    for i in range(n):
        if i not in fea:
            X=np.delete(X,i-count,axis=1)
            count+=1
    test_X = X
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    #test_X_file_path="train_X_pr.csv"
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 