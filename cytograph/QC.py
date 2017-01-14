from pylab import *
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import RANSACRegressor
from scipy.stats import t as student_t

LOW_EXPRESSION_THRESH = 1700


def low_complexity_flagger(total_molecules: np.ndarray, expressed_genes: np.ndarray) -> np.ndarray:
    """It flags low complexity cells based on the total_molecules-expressed_genes relation

    The algorythm has 2 steps that ensure robustness
    1) Fits a robust curve estiamator (SVR-RANSAC Model) to the total_molecules-expressed_genes relation
    2) From the RANSAC Model:
       A) Extract the inlier/ outlier flag of the ransac procedure
       B) Calculate the square residuals
    3) Refine the labels by modelling the residual distribution as a 2 component Student's T mixture model.
       (similar to a Expectation-Maximization step)

    Args
    ----
    total_molecules: numpy.1darray
        the sum over all the genes of the number of molecules
    expressed_genes: numpy.1darray
        the number of genes that are expressed at at least 1 molecule 

    Return
    ------
    indicator_vector: numpy.1darray(bool)
        A vector that contains True if the cell has low complexity 


    Notes
    -----
    Cells with low total molecule counts should be excluded before running this function.
    The function should be used dataset-wise (e.g. once per loom file)

    """
    # Log transform and reshape the data
    X = np.log2(total_molecules)
    y = np.log2(expressed_genes)
    X = X.reshape(-1,1)

    # Fit the model using the RANSAC procedure
    model_base = SVR(kernel="rbf", gamma=0.05)
    model = RANSACRegressor(model_base, min_samples=120, max_trials=100 )
    model.fit(X, y)

    # Calculate residuals
    delta = y - model.predict(X)
    sq_residuals = (delta)**2

    # Here we are interested in outlyers with lower complexity (e.g. below the curve)
    outlier = np.logical_not(model.inlier_mask_)
    outlier = np.logical_and(outlier, delta < 0)
    inlier = np.logical_not(outlier).astype(int)

    # Calculate the sample mean and standard deviation
    mu0, sd0 = np.mean(sq_residuals[inlier == 0]), np.std(sq_residuals[inlier == 0])
    mu1, sd1 = np.mean(sq_residuals[inlier == 1]), np.std(sq_residuals[inlier == 1])

    # Calculate the expectation of the model using the sample extimates
    mu_space = np.linspace(mu1, mu0, num=200)  # because mu1 < mu0
    w0, lik0 = np.sum(inlier == 0), student_t.pdf(loc=mu0, scale=sd0, x=mu_space, df=2)
    w1, lik1 = np.sum(inlier == 1), student_t.pdf(loc=mu1, scale=sd1, x=mu_space, df=2)

    # Find the point where it becomes more likely to be an outlyer than an inlier
    more_lik_outlyer = w0 * lik0 > w1 * lik1
    thresh = mu_space[np.where(more_lik_outlyer)[0][0]]

    return (sq_residuals > thresh).astype(bool)


def low_expression_flagger(total_molecules: float, thresh=LOW_EXPRESSION_THRESH):
    """Trivial function that to flag low cells fit low expression

    Args
    ----
    total_molecules: numpy.1darray
        the sum over all the genes of the number of molecules

    Return
    ------
    indicator_vector: numpy.1darray(bool)
        A vector that contains True if the cell has low molecule count 

    """

    return (total_molecules<thresh).astype(bool)

def doublet_flagger(df):
    """Combine several heuristic to robustly call dublets

    Args
    ----
    df: pandas.DataFrame
        The dataset

    Return
    ------
    indicator_vector: numpy.1darray(int)
        A vector that contains 1 if the cell is a dublet 

    """
    return

def doublet_heuristic_1():
    """Heuristic that score the likelihood that a data entry is a doublet
    Args
    ----

    Return
    ------

    """
    return None

def doublet_heuristic_2():
    """Heuristic that score the likelihood that a data entry is a doublet
    Args
    ----

    Return
    ------

    """
    return None

def doublet_heuristic_3():
    """Heuristic that score the likelihood that a data entry is a doublet
    Args
    ----

    Return
    ------

    """
    return None

