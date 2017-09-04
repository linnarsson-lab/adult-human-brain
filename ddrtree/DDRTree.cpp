#include "DDRTree.h"
#include <stdlib.h> // for NULL
#include <iostream>

#include <boost/functional/hash.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

using namespace Eigen;
using namespace std;

typedef Eigen::SparseMatrix<double> SpMat;


void pca_projection(const MatrixXd& C, int dimensions,  MatrixXd& W){
    EigenSolver<MatrixXd> es(C, true);

    MatrixXd eVecs = es.eigenvectors().real();
    VectorXd eVals = es.eigenvalues().real();

    // Sort by ascending eigenvalues:
    std::vector<std::pair<double,MatrixXd::Index> > D;
    D.reserve(eVals.size());
    for (MatrixXd::Index i=0;i<eVals.size();i++)
        D.push_back(std::make_pair<double,MatrixXd::Index>((double)eVals.coeff(i,0),(long)i));
    std::sort(D.rbegin(),D.rend());
    MatrixXd sortedEigs;
    sortedEigs.resize(eVecs.rows(), dimensions);
    for (int i=0; i < eVals.size() && i < dimensions; i++)
    {
        eVals.coeffRef(i,0)=D[i].first;
        sortedEigs.col(i)=eVecs.col(D[i].second);
    }
    W = sortedEigs;
}


double get_major_eigenvalue(const MatrixXd& M){
    // Construct matrix operation object using the wrapper class DenseGenMatProd
    DenseSymMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver< double, LARGEST_ALGE, DenseSymMatProd<double> > eigs(&op, 3, 6);

    // Initialize and compute
    eigs.init();
    eigs.compute();
    if(eigs.info() == SUCCESSFUL)
        return eigs.eigenvalues()[0];

    return 0;
}

void sq_dist(const MatrixXd& a, const MatrixXd& b,  MatrixXd& W){
//     aa <- colSums(a^2)
//     bb <- colSums(b^2)
//     ab <- t(a) %*% b
//
//     aa_repmat <- matrix(rep(aa, times = ncol(b)), ncol = ncol(b), byrow = F)
//     bb_repmat <- matrix(rep(bb, times = ncol(a)), nrow = ncol(a), byrow = T)
//     dist <- abs(aa_repmat + bb_repmat - 2 * ab)

//    cout << "   a nan check : (" << a.rows() << "x" << a.cols() << ", " << a.maxCoeff() << " )" << std::endl;
//    cout << "   b nan check : (" << b.rows() << "x" << b.cols() << ", " << b.maxCoeff() << " )" << std::endl;

    VectorXd aa = (a.array() * a.array()).colwise().sum();
    VectorXd bb = (b.array() * b.array()).colwise().sum();
    MatrixXd ab = a.transpose() * b;
//    cout << "   ab nan check : (" << ab.rows() << "x" << ab.cols() << ", " << ab.maxCoeff() << " )" << std::endl;

    MatrixXd aa_repmat;
    aa_repmat.resize(a.cols(), b.cols());
    for (int i=0; i < aa_repmat.cols(); i++)
    {
        aa_repmat.col(i) = aa;
    }
//    cout << "   aa_repmat nan check : (" << aa_repmat.rows() << "x" << aa_repmat.cols() << ", " << aa_repmat.maxCoeff() << " )" << std::endl;

    MatrixXd bb_repmat;
    bb_repmat.resize(a.cols(), b.cols());
    for (int i=0; i < bb_repmat.rows(); i++)
    {
        bb_repmat.row(i) = bb;
    }

//    cout << "   bb_repmat nan check : (" << bb_repmat.rows() << "x" << bb_repmat.cols() << ", " << bb_repmat.maxCoeff() << " )" << std::endl;
    W = aa_repmat + bb_repmat - 2 * ab;
//    cout << "   W nan check : (" << W.rows() << "x" << W.cols() << ", " << W.maxCoeff() << " )" << std::endl;


    W = W.array().abs().matrix();
}

void DDRTree_reduce_dim(const MatrixXd& X_in,
                            const MatrixXd& Z_in,
                            const MatrixXd& Y_in,
                            const MatrixXd& W_in,
                            int dimensions,
                            int maxIter,
                            int num_clusters,
                            double sigma,
                            double lambda,
                            double gamma,
                            double eps,
                            bool verbose,
                            MatrixXd& Y_out,
                            SpMat& stree,
                            MatrixXd& Z_out, 
                            MatrixXd& W_out,
                            std::vector<double>& objective_vals){

    Y_out = Y_in;
    W_out = W_in;
    Z_out = Z_in;

    int N_cells = X_in.cols();
/*
    typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
    typedef boost::adjacency_matrix<
                                  boost::undirectedS, boost::no_property,
                                  EdgeWeightProperty> Graph;
    typedef boost::graph_traits < Graph >::edge_descriptor Edge;
    typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;

    if (verbose)
        cout << "setting up adjacency matrix" << std::endl;
    Graph g(Y_in.cols());
    for (std::size_t j = 0; j < Y_in.cols(); ++j) {
        for (std::size_t i = 0; i < Y_in.cols() && i <= j ; ++i) {
            Edge e; bool inserted;
            tie(e, inserted) = add_edge(i, j, g);
        }
    }
*/
    using namespace boost;
    typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
    typedef boost::adjacency_list < vecS, vecS, undirectedS,
                                    property<vertex_distance_t, double>, property < edge_weight_t, double > > Graph;

    typedef boost::graph_traits < Graph >::edge_descriptor Edge;
    typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;

    Graph g(Y_in.cols());
    //property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
    for (std::size_t j = 0; j < Y_in.cols(); ++j) {
        for (std::size_t i = 0; i < Y_in.cols() && i <= j ; ++i) {
            if (i != j){
                Edge e; bool inserted;
                tie(e, inserted) = add_edge(i, j, g);
            }
        }
    }

    boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get(boost::edge_weight_t(), g);

    MatrixXd B = MatrixXd::Zero(Y_in.cols(), Y_in.cols());

    std::vector < graph_traits < Graph >::vertex_descriptor >
        old_spanning_tree(num_vertices(g));

    // std::vector<double> objective_vals;

    MatrixXd distsqMU;
    MatrixXd L;
    MatrixXd distZY;
    distZY.resize(X_in.cols(), num_clusters);

    MatrixXd min_dist;
    min_dist.resize(X_in.cols(), num_clusters);

    MatrixXd tmp_distZY;
    tmp_distZY.resize(X_in.cols(), num_clusters);

    //SpMat tmp_R(X_in.cols(), num_clusters);
    MatrixXd tmp_R;
    tmp_R.resize(X_in.cols(), num_clusters);

    //SpMat R(X_in.cols(), num_clusters);
    MatrixXd R;
    R.resize(tmp_R.rows(), num_clusters);

    //SpMat Gamma(R.cols(), R.cols());
    MatrixXd Gamma = MatrixXd::Zero(R.cols(), R.cols());

    SpMat tmp(Gamma.rows(), Gamma.cols());

    MatrixXd tmp_dense;
    tmp_dense.resize(Gamma.rows(), Gamma.cols());

    //SpMat Q;
    MatrixXd Q;
    Q.resize(tmp_dense.rows(), R.rows());

    MatrixXd C;
    C.resize(X_in.rows(), Q.cols());

    MatrixXd tmp1;
    tmp1.resize(C.rows(), X_in.rows());


    for (int iter = 0; iter < maxIter; ++iter){
        if (verbose)
            cout << "************************************** " << std::endl;
        if (verbose)
            cout << "Iteration: " << iter << std::endl;

        sq_dist(Y_out, Y_out, distsqMU);
        //cout << "distsqMU: " << distsqMU<< std::endl;
        std::pair<edge_iter, edge_iter> edgePair;
        if (verbose)
            cout << "updating weights in graph" << std::endl;
        for(edgePair = edges(g); edgePair.first != edgePair.second; ++edgePair.first)
        {
            if (source(*edgePair.first,g) != target(*edgePair.first,g)){
                //cout << "edge: " << source(*edgePair.first,g) << " " << target(*edgePair.first,g) << " : " << distsqMU(source(*edgePair.first,g), target(*edgePair.first,g)) << std::endl;
                EdgeWeightMap[*edgePair.first] = distsqMU(source(*edgePair.first,g), target(*edgePair.first,g));
            }
        }

        std::vector < graph_traits < Graph >::vertex_descriptor >
            spanning_tree(num_vertices(g));

        if (verbose)
            cout << "Finding MST" << std::endl;
        prim_minimum_spanning_tree(g, &spanning_tree[0]);

        if (verbose)
            cout << "Refreshing B matrix" << std::endl;
        // update the adjacency matrix. First, erase the old edges
        for (size_t ei = 0; ei < old_spanning_tree.size(); ++ei)
        {
//if (ei != old_spanning_tree[ei]){
                B(ei, old_spanning_tree[ei]) = 0;
                B(old_spanning_tree[ei], ei) = 0;
//            }
        }

        // now add the new edges
        for (size_t ei = 0; ei < spanning_tree.size(); ++ei)
        {
            if (ei != spanning_tree[ei]){
                B(ei, spanning_tree[ei]) = 1;
                B(spanning_tree[ei], ei) = 1;
            }
        }
        //cout << "B: " << std::endl << B << std::endl;
        if (verbose)
            cout << "   B : (" << B.rows() << " x " << B.cols() << ")" << std::endl;

        old_spanning_tree = spanning_tree;

        L = B.colwise().sum().asDiagonal();
        L = L - B;
        //cout << "   Z_out nan check : (" << Z_out.rows() << "x" << Z_out.cols() << ", " << Z_out.maxCoeff() << " )" << std::endl;

        //cout << "   Y_out nan check : (" << Y_out.rows() << "x" << Y_out.cols() << ", " << Y_out.maxCoeff() << " )" << std::endl;

        sq_dist(Z_out, Y_out, distZY);
        //cout << "   distZY nan check : (" << distZY.maxCoeff() << " )" << std::endl;
        if (verbose)
            cout << "   distZY : (" << distZY.rows() << " x " << distZY.cols() << ")" << std::endl;

        if (verbose)
            cout << "   min_dist : (" << min_dist.rows() << " x " << min_dist.cols() << ")" << std::endl;
        //min_dist <- matrix(rep(apply(distZY, 1, min), times = K), ncol = K, byrow = F)

        VectorXd distZY_minCoeff = distZY.rowwise().minCoeff();
	    if (verbose)
            cout << "distZY_minCoeff = " << std::endl;
	    for (int i=0; i < min_dist.cols(); i++)
        {
            min_dist.col(i) = distZY_minCoeff;
        }
        //cout << min_dist << std::endl;

        //tmp_distZY <- distZY - min_dist
        tmp_distZY = distZY - min_dist;
        //cout << tmp_distZY << std::endl;

        if (verbose)
            cout << "   tmp_R : (" << tmp_R.rows() << " x " << tmp_R.cols() << ")" << std::endl;
        //tmp_R <- exp(-tmp_distZY / params$sigma)
        tmp_R = tmp_distZY.array() / (-1.0 * sigma);
        //cout << tmp_R << std::endl;

        tmp_R = tmp_R.array().exp().matrix();

        if (verbose)
            cout << "   R : (" << R.rows() << " x " << R.cols() << ")" << std::endl;
        //R <- tmp_R / matrix(rep(rowSums(tmp_R), times = K), byrow = F, ncol = K)

        VectorXd tmp_R_rowsums =  tmp_R.rowwise().sum();
        for (int i=0; i < R.cols(); i++)
        {
            R.col(i) = tmp_R_rowsums;
        }
        //cout << R << std::endl;
        //cout << "&&&&&" << std::endl;
        R = (tmp_R.array() / R.array()).matrix();
        //cout << R << std::endl;

        if (verbose)
            cout << "   Gamma : (" << Gamma.rows() << " x " << Gamma.cols() << ")" << std::endl;
        //Gamma <- matrix(rep(0, ncol(R) ^ 2), nrow = ncol(R))
        Gamma = MatrixXd::Zero(R.cols(), R.cols());
        //diag(Gamma) <- colSums(R)
        Gamma.diagonal() = R.colwise().sum();
        //cout << Gamma << std::endl;

        //termination condition
        //obj1 <- - params$sigma * sum(log(rowSums(exp(-tmp_distZY / params$sigma))) - min_dist[, 1] / params$sigma)
        VectorXd x1 = (tmp_distZY.array() / -sigma).exp().rowwise().sum().log();
        //cout << "Computing x1 " << x1.transpose() << std::endl;
        double obj1 = -sigma * (x1 - min_dist.col(0) / sigma).sum();
        //cout << obj1 << std::endl;

        //obj2 <- (norm(X - W %*% Z, '2'))^2 + params$lambda * sum(diag(Y %*% L %*% t(Y))) + params$gamma * obj1 #sum(diag(A))
        //cout << X_in - W_out * Z_out << std::endl;

        if (verbose){
            cout << "   X : (" << X_in.rows() << " x " << X_in.cols() << ")" << std::endl;
            cout << "   W : (" << W_out.rows() << " x " << W_out.cols() << ")" << std::endl;
            cout << "   Z : (" << Z_out.rows() << " x " << Z_out.cols() << ")" << std::endl;
        }
        double major_eigen_value = get_major_eigenvalue(X_in - W_out * Z_out);
        double obj2 = major_eigen_value;
        //cout << "norm = " << obj2 << std::endl;
        obj2 = obj2 * obj2;

        if (verbose){
            cout << "   L : (" << L.rows() << " x " << L.cols() << ")" << std::endl;
        }

        obj2 = obj2 + lambda * (Y_out * L * Y_out.transpose()).diagonal().sum() + gamma * obj1;
        //cout << obj2 << std::endl;
        //cout << "obj2 = " << obj2 << std::endl;
        objective_vals.push_back(obj2);

        if (verbose)
            cout << "Checking termination criterion" << std::endl;
        if(iter >= 1) {
            double delta_obj = std::abs(objective_vals[iter] - objective_vals[iter - 1]);
            delta_obj /=  std::abs(objective_vals[iter - 1]);
            if (verbose)
                cout << "delta_obj: " << delta_obj << std::endl;
            if(delta_obj < eps) {
                break;
            }
        }

        //cout << "L" << std::endl;
        //cout << L << std::endl;
        if (verbose)
            cout << "Computing tmp" << std::endl;
        //tmp <- t(solve( ( ( (params$gamma + 1) / params$gamma) * ((params$lambda / params$gamma) * L + Gamma) - t(R) %*% R), t(R)))

        if (verbose)
            cout << "... stage 1" << std::endl;
        tmp = ((Gamma + (L * (lambda / gamma))) * ((gamma + 1.0) / gamma)).sparseView();
        //cout << tmp << std::endl;
        if (verbose){
            cout << "... stage 2" << std::endl;
       	    //cout << R.transpose() << std::endl;
	    }

	    SparseMatrix<double> R_sp = R.sparseView();
	    tmp = tmp - (R_sp.transpose() * R_sp);
        //tmp = tmp_dense.sparseView();

        if (verbose){
            cout << "Pre-computing LLT analysis" << std::endl;
            cout << "tmp is (" << tmp.rows() << "x" << tmp.cols() <<"), " << tmp.nonZeros() << " non-zero values" << std::endl;
        }

        //cout << tmp << std::endl;
        SimplicialLLT <SparseMatrix<double>, Lower, AMDOrdering<int> > solver;
        solver.compute(tmp);
        if(solver.info()!=Success) {
            // decomposition failed
            cout << "Error!" << std::endl;
            tmp_dense = tmp;
            tmp_dense = tmp_dense.partialPivLu().solve(R.transpose()).transpose();
            cout << tmp_dense << std::endl;
        }else{
            if (verbose)
                cout << "Computing LLT" << std::endl;
            tmp_dense = solver.solve(R.transpose()).transpose();
            if(solver.info()!=Success) {
                // solving failed
                cout << "Error!" << std::endl;
            }
        }

        //tmp_dense = tmp_dense.llt().solve(R.transpose()).transpose();
        if (verbose)
            cout << "tmp_dense " << tmp_dense.rows() << "x" << tmp_dense.cols() <<") "<< std::endl;

        if (verbose)
            cout << "Computing Q " << Q.rows() << "x" << Q.cols() <<") "<< std::endl;
        //Q <- 1 / (params$gamma + 1) * (diag(1, N) + tmp %*% t(R))
        //tmp = tmp_dense.sparseView();
        //cout << "tmp_dense is (" << tmp_dense.rows() << "x" << tmp_dense.cols() <<"), " << tmp_dense.nonZeros() << " non-zero values" << std::endl;
        //cout << "R_sp is (" << R_sp.rows() << "x" << R_sp.cols() <<"), " << R_sp.nonZeros() << " non-zero values" << std::endl;

        /////////////////////////
        /*
        double gamma_coeff = 1.0 / (1 + gamma);

        SpMat Q_id(tmp_dense.rows(), R.rows());
        Q_id.setIdentity();

        tmp1 = gamma_coeff * (X_in * tmp_dense.sparseView());
        if (verbose)
            cout << "First tmp1 product complete: " << tmp1.rows() << "x" << tmp1.cols() <<"), " << tmp1.nonZeros() << " non-zero values" << std::endl;
        tmp1 = tmp1 * R_sp.transpose();
        if (verbose)
            cout << "Second tmp1 product complete: " << tmp1.rows() << "x" << tmp1.cols() <<"), " << tmp1.nonZeros() << " non-zero values" << std::endl;
        tmp1 += gamma_coeff * X_in;
        if (verbose)
            cout << "Third tmp1 product complete: " << tmp1.rows() << "x" << tmp1.cols() <<"), " << tmp1.nonZeros() << " non-zero values" << std::endl;
        tmp1 = tmp1 * X_in.transpose();
        if (verbose)
            cout << "Final tmp1 product complete: " << tmp1.rows() << "x" << tmp1.cols() <<"), " << tmp1.nonZeros() << " non-zero values" << std::endl;
        */
        ///////////////////////////
/*
         Q = ((MatrixXd::Identity(X_in.cols(), X_in.cols()) + (tmp_dense * R.transpose()) ).array() / (gamma + 1.0));

         if (verbose){
        cout << "gamma: " << gamma << std::endl;
        cout << "   X_in : (" << X_in.rows() << " x " << X_in.cols() << ")" << std::endl;
        cout << "   Q : (" << Q.rows() << " x " << Q.cols() << ")" << std::endl;
        //cout << Q << std::endl;
         }

        // C <- X %*% Q
        C = X_in * Q;
        if (verbose)
        cout << "   C : (" << C.rows() << " x " << C.cols() << ")" << std::endl;
        tmp1 =  C * X_in.transpose();
*/
        /////////////////////////

        Q = (X_in + ((X_in * tmp_dense) * R.transpose()) ).array() / (gamma + 1.0);

        if (verbose){
            cout << "gamma: " << gamma << std::endl;
            cout << "   X_in : (" << X_in.rows() << " x " << X_in.cols() << ")" << std::endl;
            cout << "   Q : (" << Q.rows() << " x " << Q.cols() << ")" << std::endl;
            //cout << Q << std::endl;
        }

        // C <- X %*% Q
        //C = X_in * Q;
        C = Q;

        tmp1 =  Q * X_in.transpose();

        /////////////////////////

        //cout << tmp1 << std::endl;

        //cout << tmp1 << std::endl;

        if (verbose){
            cout << "Computing W" << std::endl;
            //cout << "tmp1 = " << std::endl;
            //cout << tmp1 << std::endl;
            //cout << (tmp1 + tmp1.transpose()) / 2 << std::endl;
        }

        //W <- pca_projection_R((tmp1 + t(tmp1)) / 2, params$dim)

        //NumericMatrix W_R = pca_projection_R((tmp1 + tmp1.transpose()) / 2,dimensions);
        //const int X_n = W_R.nrow(), X_p = W_R.ncol();
        //Map<MatrixXd> W(W_R.begin(), X_n, X_p);

        //W_out = W;
        pca_projection((tmp1 + tmp1.transpose()) / 2, dimensions, W_out);
        //cout << W_out << std::endl;

        if (verbose)
            cout << "Computing Z" << std::endl;
        //Z <- t(W) %*% C
        Z_out = W_out.transpose() * C;
        //cout << Z_out << std::endl;

        if (verbose)
            cout << "Computing Y" << std::endl;
        //Y <- t(solve((params$lambda / params$gamma * L + Gamma), t(Z %*% R)))
        Y_out = L * (lambda / gamma) + Gamma;
        Y_out = Y_out.llt().solve((Z_out * R).transpose()).transpose();

        //cout << Y_out << std::endl;
    }

    if (verbose)
        cout << "Clearing MST sparse matrix" << std::endl;
    stree.setZero();

    if (verbose){
        cout << "Setting up MST sparse matrix with " << old_spanning_tree.size() << std::endl;
    }
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(2*old_spanning_tree.size());
    // Send back the weighted MST as a sparse matrix
    for (size_t ei = 0; ei < old_spanning_tree.size(); ++ei)
    {
        //stree.insert(source(*ei, g), target(*ei, g)) = 1;//distsqMU(source(*ei, g), target(*ei, g));
        tripletList.push_back(T( ei, old_spanning_tree[ei], distsqMU(ei, old_spanning_tree[ei])));
        tripletList.push_back(T( old_spanning_tree[ei], ei, distsqMU(old_spanning_tree[ei], ei)));
    }
    stree = SpMat(N_cells, N_cells);
    stree.setFromTriplets(tripletList.begin(), tripletList.end());
}


