'''
Shows the use of LDA v/s QDA for any data.
Change the means and cov in main() to get new data

Useage :
    python trial_lda.py generate
    python trial_lda.py run
'''

# import seaborn as sns
import matplotlib.mlab as mlab
from matplotlib import rc
from pylab import axes
from matplotlib.widgets import Slider
from mlxtend.evaluate import plot_decision_regions
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

plt.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})


def generate(mean1, cov1, n_pts):
    return np.random.multivariate_normal(mean=mean_1,
                                         cov=cov_1,
                                         size=(n_pts, ))


def print_data(lda_first, qda_first, X_test, y_test):
    print("LDA")
    print('Prediction accuracy is : ' + str(lda_first.score(X_test, y_test) *
                                            100) + ' %')
    print("Means are : ")
    print(lda_first.means_[0])
    print(lda_first.means_[1])
    print("Covariance are : ")
    print(lda_first.covariance_)

    print("\n--------------****************------------------\n")

    print("QDA")
    print('Prediction accuracy is : ' + str(qda_first.score(X_test, y_test) *
                                            100) + ' %')
    print("Means are : ")
    print(qda_first.means_[0])
    print(qda_first.means_[1])
    print("Covariance are : ")
    print(qda_first.covariances_[0])
    print(qda_first.covariances_[1])


def PLOT_DB(lda_first, qda_first, X, y, X_test, y_test):

    plt.figure(1)

    # Plot the LDA
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(bottom=0.25, right=0.75)
    plot_decision_regions(X=X,
                          y=y,
                          clf=lda_first,
                          colors='limegreen,red',
                          legend=2)
    title_string1 = "Linear Decision Boundry \nAccuracy is : " + str(str(
        lda_first.score(X_test, y_test) * 100) + ' %')
    plt.title(title_string1)
    plt.scatter(lda_first.means_[0][0], qda_first.means_[0][1], c='blue',
                marker='+', linewidth='5', s=180, label="Class 0 mean")
    plt.scatter(lda_first.means_[1][0], qda_first.means_[1][1], c='yellow',
                marker='+', linewidth='5', s=180, label="Class 1 mean")
    plt.xlabel("Variable " + '$x_1$')
    plt.ylabel("Variable " + '$x_2$')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.subplots_adjust(bottom=0.25, right=0.75)
    plot_decision_regions(X=X,
                          y=y,
                          clf=qda_first,
                          colors='limegreen,red',
                          legend=2)
    plt.title("Quadratic Decision Boundry \nAccuracy is : " + str(str(
        qda_first.score(X_test, y_test) * 100) + ' %'))
    plt.scatter(qda_first.means_[0][0], qda_first.means_[0][1], c='blue',
                marker='+', linewidth='5', s=180, label="Class 0 mean")
    plt.scatter(qda_first.means_[1][0], qda_first.means_[1][1], c='yellow',
                marker='+', linewidth='5', s=180, label="Class 1 mean")
    plt.xlabel("Variable " + '$x_1$')
    plt.ylabel("Variable " + "$x_2$")
    plt.legend(loc='upper left')

    plt.tight_layout()


def main():

    if (sys.argv[1] == 'generate'):

        n_pts1 = 100 ;n_pts2 = 1000

        # Data parameters
        mean_1 = [0, 0]
        cov_1 = np.array([[1, 0], [0, 1]])
        mean_2 = [3, -1]
        cov_2 = np.array([[1, 0], [0, 5]])

        # Create and save data
        val_1 = np.random.multivariate_normal(mean=mean_1,
                                              cov=cov_1,
                                              size=(n_pts1, ))
        val_2 = np.random.multivariate_normal(mean=mean_2,
                                              cov=cov_2,
                                              size=(n_pts2, ))

        label_1 = np.array(([0] * n_pts1), dtype=int)
        label_2 = np.array(([1] * n_pts2), dtype=int)

        np.savetxt("./data1.txt", val_1)
        np.savetxt("./data2.txt", val_2)
        np.savetxt("./data1_label.txt", label_1)
        np.savetxt("./data2_label.txt", label_2)

        print val_1
        print val_2

        sys.exit("Exiting!")

    elif (sys.argv[1] == 'run'):

        # Load data from file
        val_1 = np.loadtxt("./data1.txt")
        val_2 = np.loadtxt("./data2.txt")
        label_1 = np.loadtxt("./data1_label.txt")
        label_2 = np.loadtxt("./data2_label.txt")

        # Get data in requred form
        X = np.vstack((val_1, val_2))
        y = np.hstack((label_1, label_2))
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.3)

        # Start training
        lda_first = LinearDiscriminantAnalysis(solver='svd',
                                               store_covariance=True)
        qda_first = QuadraticDiscriminantAnalysis(store_covariances=True)
        lda_first.fit(X=X_train, y=y_train)
        qda_first.fit(X=X_train, y=y_train)

        # Print the variables
        print_data(lda_first, qda_first, X_test, y_test)

        # Plot decision boundry
        PLOT_DB(lda_first, qda_first, X, y, X_test, y_test)

        plt.show()


if __name__ == '__main__':
    main()
