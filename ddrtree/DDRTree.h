#ifndef _DDRTree_DDRTREE_H
#define _DDRTree_DDRTREE_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Spectra/SymEigsSolver.h"
using namespace Eigen;
using namespace Spectra;

void pca_projection_cpp(const MatrixXd& R_C, int dimensions,  MatrixXd& W);
void sq_dist_cpp(const MatrixXd& a, const MatrixXd& b,  MatrixXd& W);

#endif
