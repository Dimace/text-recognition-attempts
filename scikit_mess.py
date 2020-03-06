from sklearn.model_selection import train_test_split
X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
print('X_train ' + str(X_train))

Y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y_train, Y_test = train_test_split(X, test_size=0.2, random_state=0)
print('Y_train ' + str(Y_train))