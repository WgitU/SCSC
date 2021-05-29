#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <Rmath.h>
#include <omp.h>

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends("RcppArmadillo")]]

void update_theta(arma::mat &theta_t, const arma::mat &y_t, const arma::mat &mu_t, const arma::mat &beta_t, 
                  const arma::uvec &clas_cell_t, const arma::vec &sigma_sq_t, const arma::vec &scale_factor_est, 
                  const arma::uvec &S_vec_t, const int &G, 
                  const int &total_cell_num, const double &tau_theta, const int &n_threads) {
  int g;
  
  arma::mat theta_star = theta_t + randn(total_cell_num, G)*tau_theta;
  arma::mat tmp = randu(total_cell_num, G);
  
#pragma omp parallel shared(theta_t, theta_star, tmp) private(g) num_threads(n_threads)
{
#pragma omp for schedule(auto)
  for (g = 0; g < G; g++) {  
    arma::vec theta_star_g = theta_star.col(g);
    arma::vec theta_g = theta_t.col(g);
    arma::vec tmp_g = tmp.col(g);
    
    //calculate the acceptance probability
    arma::vec part1 = - (scale_factor_est % (exp(theta_star_g) - exp(theta_g)));
    
    arma::vec part2 = y_t.col(g) % (theta_star_g - theta_g);
    
    arma::vec mu_t_g = mu_t.row(g).t();
    arma::vec beta_t_g = beta_t.row(g).t();
    arma::vec part3 = -(theta_star_g-theta_g)%(theta_star_g+theta_g-2.0*(mu_t_g(clas_cell_t)+
      beta_t_g(S_vec_t)))/(2.0*sigma_sq_t(g));
    
    arma::vec log_r = part1 + part2 + part3;
    arma::vec r = exp(log_r);
    //update theta_t for gene g 
    arma::uvec ind = arma::find(tmp_g < r);
    theta_g(ind) = theta_star_g(ind);
    theta_t.col(g) = theta_g;
  }
}
}


arma::mat l_factorial(arma::mat x) {
  int x_row = x.n_rows;
  int x_col = x.n_cols;
  arma::mat fa(x_row, x_col);
  for (int i = 0; i < x_row; i++) {
    for (int t = 0; t < x_col; t++) {
      fa(i,t) = Rcpp::internal::lfactorial(x(i,t));
    }
  }
  return fa;
}

void update_y(arma::mat &y_t, const arma::mat &theta_t, const arma::mat &ind_zero, const arma::vec &scale_factor_est,
              const arma::vec &lambda0_t, const arma::vec &lambda1_t, const int &G, const int &total_cell_num, 
              const int &radius, const int &n_threads) {
  int g;
  
  arma::vec rad = arma::regspace(-radius, 1, radius);
  arma::mat y_star = y_t;
  arma::uvec ind_0 = arma::find(ind_zero == true);
  y_star(ind_0) = y_t(ind_0) + Rcpp::RcppArmadillo::sample(rad, ind_0.n_elem, true);
  
  arma::mat tmp_unif = randu(total_cell_num, G);
  
#pragma omp parallel shared(y_t, y_star, tmp_unif) private(g) num_threads(n_threads)
{
#pragma omp for schedule(auto)
  for (g = 0; g < G; g++) {
    arma::vec y_g = y_t.col(g);
    arma::vec theta_g = theta_t.col(g);
    arma::vec y_star_g = y_star.col(g);
    arma::uvec ind_0 = arma::find(ind_zero.col(g) == true);
    int ind_nzero = ind_0.n_elem;
    if (ind_nzero > 0) {
      
      //calculate the acceptance probability for subject j
      arma::vec r(total_cell_num, fill::zeros);
      
      arma::vec ind_zero_g = ind_zero.col(g);
      arma::uvec ind_tmp = arma::find((y_star_g >= 0) && (ind_zero_g == true) && (y_g >= 0));
      arma::uvec ind_0_1 = arma::find((y_star_g >= 0) && (ind_zero_g == true) && (y_g > 0));
      arma::uvec ind_1_0 = arma::find((y_star_g > 0) && (ind_zero_g == true) && (y_g >= 0));
      
      if (ind_tmp.n_elem > 0) {
        arma::vec part1 = (theta_g(ind_tmp) + log(scale_factor_est(ind_tmp)))%(y_star_g(ind_tmp) - y_g(ind_tmp));
        arma::vec part2 = l_factorial(y_g(ind_tmp)) - l_factorial(y_star_g(ind_tmp));
        r(ind_tmp) = exp(part1 + part2);
      }
      
      if (ind_0_1.n_elem > 0) {
        arma::vec part3 = arma::normcdf(lambda0_t(g) + lambda1_t(g) * log2(y_g(ind_0_1)+1.0));
        r(ind_0_1) = r(ind_0_1)/part3;
      }
      
      if (ind_1_0.n_elem > 0) {
        arma::vec part4 = arma::normcdf(lambda0_t(g) + lambda1_t(g) * log2(y_star_g(ind_1_0)+1.0));
        r(ind_1_0) = r(ind_1_0)%part4;
      }
      
      //update y_t for subject j
      arma::vec tmp_unif_g = tmp_unif.col(g);
      
      arma::uvec ind = arma::find(tmp_unif_g < r);
      if (ind.n_elem > 0) {
        y_g(ind) = y_star_g(ind);
        y_t.col(g) = y_g;
      }
    }
  }
}
}


void update_mu(arma::mat &mu_t, const arma::mat &theta_t, const arma::mat &beta_t, const arma::vec &sigma_sq_t, 
               const arma::uvec &clas_cell_t, const arma::uvec &S_vec_t, 
               const int &K, const int &G, const double &eta_mu, const double &tau_sq_mu) {
  for (int k = 0; k < K; k++) {
    arma::vec tmp1(G, fill::zeros);
    double tmp2 = 0;
    
    arma::uvec ind_k = arma::find(clas_cell_t == k);
    if (ind_k.n_elem > 0) {
      arma::uvec S_k = S_vec_t(ind_k);
      tmp1 = arma::sum(theta_t.rows(ind_k).t() - beta_t.cols(S_k), 1);
      tmp2 = ind_k.n_elem;
    }
    
    arma::vec sd = 1.0/(tmp2/sigma_sq_t + 1.0/tau_sq_mu);
    arma::vec avg = (tmp1/sigma_sq_t + eta_mu/tau_sq_mu) % sd;
    sd = sqrt(sd);
    
    for (int i = 0; i < G; i++){
      mu_t(i,k) = R::rnorm(avg(i), sd(i));
    }
  }
}

void update_beta(arma::mat &beta_t, const arma::mat &theta_t, const arma::mat &mu_t, const arma::vec &sigma_sq_t, 
                 const arma::uvec &clas_cell_t, const arma::uvec &S_vec_t,
                 const int &L, const int &G, const double &eta_beta, const double &tau_sq_beta) {
  for (int ell = 0; ell < L; ell++) {
    arma::vec tmp1(G, fill::zeros);
    double tmp2 = 0;
    
    arma::uvec ind_ell = arma::find(S_vec_t == ell);
    if (ind_ell.n_elem > 0) {
      arma::uvec clas_cell_ell = clas_cell_t(ind_ell);
      tmp1 = arma::sum(theta_t.rows(ind_ell).t() - mu_t.cols(clas_cell_ell), 1);
      tmp2 = ind_ell.n_elem;
    }
    
    arma::vec sd = 1.0/(tmp2/sigma_sq_t + 1.0/tau_sq_beta);
    arma::vec avg = (tmp1/sigma_sq_t + eta_beta/tau_sq_beta) % sd;
    sd = sqrt(sd);
    
    for (int i = 0; i < G; i++){
      beta_t(i,ell) = R::rnorm(avg(i), sd(i));
    }
  }
}


void update_mu_prior(double &eta_mu, double &tau_sq_mu, const arma::mat &mu, const int &K, const int &G) {
  double mu_var =  tau_sq_mu/(G*K);
  double mu_mean = arma::accu(mu)/(G*K);
  double eta_mu_star = R::rnorm(mu_mean, sqrt(mu_var));
  
  double shape = (G*K - 1.0)/2.0;
  double scale = 2.0/arma::accu(pow(mu-eta_mu,2));
  double tau_sq_mu_star = 1.0/arma::randg(distr_param(shape,scale));
  
  eta_mu = eta_mu_star;
  tau_sq_mu = tau_sq_mu_star;
}


void update_beta_prior(double &eta_beta, double &tau_sq_beta, arma::mat beta, const int &L, const int &G, const int &S_0) {
  double beta_var = tau_sq_beta/(G*(L-1.0));
  beta.shed_col(S_0);
  double beta_mean = arma::accu(beta)/(G*(L-1.0));
  double eta_beta_star = R::rnorm(beta_mean, sqrt(beta_var));
  
  double shape = (G*(L-1.0) - 1.0)/2.0;
  double scale = 2.0/arma::accu(pow(beta-eta_beta,2));
  double tau_sq_beta_star = 1.0/arma::randg(distr_param(shape,scale));
  
  eta_beta = eta_beta_star;
  tau_sq_beta = tau_sq_beta_star;
}


void update_sigma_sq(arma::vec &sigma_sq_t, const arma::mat &theta_t, const arma::mat &mu_t, const arma::mat &beta_t, 
                     const arma::uvec &clas_cell_t, const arma::uvec &S_vec_t, const int &G, const int &total_cell_num, const int &n_threads) {
  double shape = total_cell_num/2.0;
  arma::vec scale_vec(G, fill::zeros);
  int g;
#pragma omp parallel shared(scale_vec) private(g) num_threads(n_threads)
{
#pragma omp for schedule(auto)
  for (g = 0; g < G; g++) {
    arma::vec theta_g = theta_t.col(g);
    arma::vec mu_g = mu_t.row(g).t();
    arma::vec beta_g = beta_t.row(g).t();
    double tmp = arma::sum(pow(theta_g - mu_g(clas_cell_t) - beta_g(S_vec_t), 2))/2.0;
    scale_vec(g) = 1.0/tmp;
  }
}
for (g = 0; g < G; g++) {
  double rgam = arma::randg(distr_param(shape,scale_vec(g)));
  sigma_sq_t(g) = 1.0/rgam;
}
}



double ldnorm(const arma::vec &x, const arma::vec &mu, const arma::vec &sigma){
  double y = arma::sum(-(x-mu)%(x-mu)/2.0/sigma/sigma - log(sigma * datum::sqrt2pi));
  return y;
}


void update_clas(arma::uvec &S_t, arma::uvec &S_vec_t, arma::uvec &clas_cell_t, const arma::mat &theta_t, const arma::mat &mu_t, const arma::mat &beta_t, const arma::mat &Pi_t,
                 const arma::uvec &ind_stat_cell, const arma::vec &sigma_sq_t, const arma::vec &phi_t, const arma::vec &n_cell,
                 const int &G, const int &total_cell_num, const int &L, const int &K, const int &m, const int &n_threads) {
  arma::vec sigma_t = sqrt(sigma_sq_t);
  int j;
  arma::uvec phi_n_zero = arma::find(phi_t > 0);
  
  arma::vec S_range = regspace(0, 1, L-1);
  arma::vec Cell_range = regspace(0, 1, K-1);
  
  arma::mat prob_S_mat(L, m);
  arma::mat prob_Cell_mat(K, total_cell_num);
  
#pragma omp parallel shared(S_t, S_vec_t, sigma_t, clas_cell_t, phi_n_zero, S_range, Cell_range, prob_S_mat, prob_Cell_mat) private(j) num_threads(n_threads)
{
#pragma omp for schedule(auto) 
  for (j = 0; j < m; j++) {
    arma::mat theta_j = theta_t.rows((ind_stat_cell(j)-1), (ind_stat_cell(j) + n_cell(j) - 2));
    arma::vec tmp(L);
    tmp.fill(- datum::inf);
    for (unsigned int ell = 0; ell < phi_n_zero.n_elem; ell++) {
      double tmp2 = 0;
      arma::vec Pi_ell = Pi_t.col(phi_n_zero(ell));
      arma::uvec Pi_ell_n_zero = arma::find(Pi_ell > 0);
      for (int i = 0; i < n_cell(j); i++) {
        arma::vec tmp1(Pi_ell_n_zero.n_elem, fill::zeros);
        for (unsigned int k = 0; k < Pi_ell_n_zero.n_elem; k++) {
          tmp1(k) = ldnorm(theta_j.row(i).t(), mu_t.col(Pi_ell_n_zero(k))+beta_t.col(phi_n_zero(ell)), sigma_t);
        }
        double tmp1_max = max(tmp1);
        tmp1 = tmp1 - tmp1_max;
        tmp2 += log(arma::sum(Pi_ell(Pi_ell_n_zero)%exp(tmp1))) + tmp1_max;
      }
      tmp(phi_n_zero(ell)) = tmp2;
    }
    arma::vec tmp_new = tmp - max(tmp);
    tmp_new = exp(tmp_new);
    arma::vec prob_S = tmp_new % phi_t;
    prob_S = prob_S / arma::sum(prob_S);
    prob_S_mat.col(j) = prob_S;
  }
  
#pragma omp critical
{
  for (j = 0; j < m; j++) {
    arma::vec prob_S_j = prob_S_mat.col(j);
    S_t(j) = Rcpp::RcppArmadillo::sample(S_range, 1, false, prob_S_j)(0);
  }
}

#pragma omp for schedule(auto) 
for (j = 0; j < m; j++) {
  S_vec_t.subvec((ind_stat_cell(j)-1), (ind_stat_cell(j) + n_cell(j) - 2)).fill(S_t(j));
  
  arma::mat theta_j = theta_t.rows((ind_stat_cell(j)-1), (ind_stat_cell(j) + n_cell(j) - 2));
  arma::uvec Pi_S_n_zero = arma::find(Pi_t.col(S_t(j)) > 0);
  for (int i = 0; i < n_cell(j); i++) {
    arma::vec tmp(K);
    tmp.fill(- datum::inf);
    for (unsigned int k = 0; k < Pi_S_n_zero.n_elem; k++) {
      tmp(Pi_S_n_zero(k)) = ldnorm(theta_j.row(i).t(), mu_t.col(Pi_S_n_zero(k))+beta_t.col(S_t(j)), sigma_t);
    }
    arma::vec tmp_new = tmp - max(tmp);
    tmp_new = exp(tmp_new);
    arma::vec prob_clas = tmp_new % Pi_t.col(S_t(j));
    prob_clas = prob_clas / arma::sum(prob_clas);
    prob_Cell_mat.col(ind_stat_cell(j) + i - 1) = prob_clas;
  }
}

#pragma omp critical
{
  for (j = 0; j < m; j++) {
    for (int i = 0; i < n_cell(j); i++) {
      arma::vec prob_cell = prob_Cell_mat.col(ind_stat_cell(j)-1+i);
      clas_cell_t(ind_stat_cell(j)-1+i) = Rcpp::RcppArmadillo::sample(Cell_range, 1, false, prob_cell)(0);
    }
  }
}
}
}



void update_phi(arma::vec &phi_t, const arma::uvec &S_t, const double &nu_t, const int &L) {
  arma::vec tmp(L, fill::none);
  for (int ell = 0; ell < L; ell++) {
    arma::uvec ind = arma::find(S_t == ell);
    tmp(ell) = ind.n_elem;
  }
  
  //update phi_t_prime
  arma::vec phi_t_prime(L, fill::ones);
  for(int ell = 0; ell < L-1; ell++) {
    double rbeta_tmp = rbeta(1,1.0+tmp(ell), nu_t+arma::sum(tmp.tail(L-ell-1)))(0);
    if (rbeta_tmp == 1){
      rbeta_tmp = rbeta_tmp - pow(10,-5);
    }
    phi_t_prime(ell) = rbeta_tmp;
  }
  
  
  //update phi_t
  phi_t(0) = phi_t_prime(0);
  
  for(int ell = 1; ell < L; ell++) {
    phi_t(ell) = (1.0-arma::sum(phi_t.head(ell)))*phi_t_prime(ell);
  }
}


arma::vec rDirichlet(const arma::vec &alpha_vec) {
  arma::vec tmp(alpha_vec.n_elem, fill::zeros);
  for (unsigned int i = 0; i < alpha_vec.n_elem; i++) {
    tmp(i) = log(arma::randg(distr_param(alpha_vec(i), 1.0)));
  }
  tmp = tmp - max(tmp);
  tmp = exp(tmp);
  tmp = tmp/arma::sum(tmp);
  return tmp;
}


void update_Pi(arma::mat &Pi_t, const arma::uvec &clas_cell_t, const arma::uvec &S_vec_t, const arma::vec &xi_vec_t, 
               const double &gam, const int &L, const int &K) {
  arma::vec xi_gamma = xi_vec_t * gam;
  for (int ell = 0; ell < L; ell++) {
    arma::vec tmp0(K, fill::zeros);
    arma::uvec ind_ell = arma::find(S_vec_t == ell);
    if (ind_ell.n_elem > 0) {
      arma::uvec clas_cell_ell = clas_cell_t(ind_ell);
      for (int k = 0; k < K; k++) {
        arma::uvec ind_j = arma::find(clas_cell_ell == k);
        tmp0(k) = ind_j.n_elem;
      }
    }
    Pi_t.col(ell) = rDirichlet(tmp0+xi_gamma);
  }
}


void update_xi_alpha(arma::vec &xi, arma::vec &xi_prime, double &alpha, const arma::mat &Pi_t, const double &gam,
                     const int &K, const int &L, const double &a_xi, const double &b_xi) {
  //update xi_prime
  arma::vec xi_prime_star_tmp = rbeta(K-1, a_xi, b_xi);
  arma::vec xi_prime_star(K, fill::none);
  xi_prime_star.head(K-1) = xi_prime_star_tmp;
  xi_prime_star(K-1) = 1.0;
  
  //update xi_t
  arma::vec xi_star(K, fill::none);
  xi_star(0) = xi_prime_star(0);
  
  for(int k = 1; k < K; k++) {
    xi_star(k) = (1.0-arma::sum(xi_star.head(k)))*xi_prime_star(k);
  }
  
  double part1 = arma::sum((a_xi - 1.0) * (log(xi_prime_star_tmp) - log(xi_prime.head(K-1)))+ 
                           (alpha + b_xi - 2.0) * (log(1.0-xi_prime_star_tmp) - log(1.0-xi_prime.head(K-1))));
  double part2 = L * arma::sum(arma::lgamma(gam * xi) - arma::lgamma(gam * xi_star));
  double part3 = gam * arma::sum((xi_star - xi) % arma::sum(log(Pi_t), 1));
  
  double log_r = part1 + part2 + part3;
  
  double r = arma::randu();
  if (r < exp(log_r)) {
    xi_prime =  xi_prime_star;
    xi = xi_star;
  }
  
  //update alpha
  double scale = -1.0/arma::sum(log(1-xi_prime.head(K-1)));
  double shape = K;
  alpha = arma::randg(distr_param(shape, scale));
}


void update_lambda(arma::vec &lambda0_t, arma::vec &lambda1_t, const arma::mat &y_t,  const arma::mat &ind_zero, 
                   const int &G, const double &lam0_0, 
                   const double &lam1_0, const double &Sigma2_lam0, 
                   const double &Sigma2_lam1, const double &sd_lambda0, 
                   const double &sd_lambda1, const int &n_threads) {
  int g;
  
  arma::vec lambda1_star = lambda1_t + sd_lambda1*randn(G);
  arma::vec lambda0_star = lambda0_t + sd_lambda0*randn(G);
  arma::vec tmp = randu(G);
  
#pragma omp parallel shared(lambda0_t, lambda1_t, lambda1_star, lambda0_star, tmp) private(g) num_threads(n_threads)
{
#pragma omp for schedule(auto)
  for (g = 0; g < G; g++) {
    double lambda1_g_star = lambda1_star(g);
    if (lambda1_g_star < 0) {
      double lambda0_g_star = lambda0_star(g);
      arma::vec log2_y_g = log2(y_t.col(g) + 1);
      arma::uvec ind_x0_yn0 = arma::find(ind_zero.col(g) == true && y_t.col(g) > 0);
      arma::uvec ind_n0 = arma::find(ind_zero.col(g) == false);
      
      arma::vec part1 = arma::normcdf(lambda0_g_star + lambda1_g_star * log2_y_g(ind_x0_yn0)) /
        arma::normcdf(lambda0_t(g) + lambda1_t(g) * log2_y_g(ind_x0_yn0));
      arma::vec part2 = (1 - arma::normcdf(lambda0_g_star + lambda1_g_star * log2_y_g(ind_n0))) /
        (1 - arma::normcdf(lambda0_t(g) + lambda1_t(g) * log2_y_g(ind_n0)));
      double part3 = exp(-((lambda0_g_star - lambda0_t(g)) * (lambda0_g_star + lambda0_t(g) - 2.0*lam0_0) / Sigma2_lam0 +
                         (lambda1_g_star - lambda1_t(g)) * (lambda1_g_star + lambda1_t(g) - 2.0*lam1_0) / Sigma2_lam1)/2.0);
      
      double r = arma::prod(part1) * arma::prod(part2) * part3;
      
      double tmp_g = tmp(g);
      if (tmp_g < r) {
        lambda0_t(g) = lambda0_g_star;
        lambda1_t(g) = lambda1_g_star;
      }
    }
  }
}
}


// [[Rcpp::export]]
List MCMC_full(const int n_iter, const int n_save, arma::mat theta_t, arma::mat y_t, arma::mat ind_zero, arma::mat beta_t, arma::mat mu_t, arma::mat Pi_t,
               arma::uvec ind_stat_cell, arma::vec phi_t, arma::vec n_cell, arma::vec sigma_sq_t, arma::uvec clas_cell_t, arma::vec lambda0_t, arma::vec lambda1_t,
               arma::uvec S_t, arma::uvec S_vec_t, arma::vec xi_vec_t, arma::vec xi_prime_t, const arma::vec scale_factor_est,
               double alpha_t, double nu_t, double gam_t, double eta_mu, double tau_sq_mu, double eta_beta, double tau_sq_beta,
               const int G, const int total_cell_num, const int L, const int K, const int m, 
               double tau_theta, int radius, double lam0_0, double lam1_0, double Sigma2_lam0, 
               double Sigma2_lam1, double sd_lambda0, double sd_lambda1, double a_xi, double b_xi,
               bool iter_save = false, int n_threads = 1, int iter_print = 1000, bool class_print = false) {
  S_t = S_t - 1;
  S_vec_t = S_vec_t - 1;
  clas_cell_t = clas_cell_t - 1;
  
  int save_start = n_iter - n_save;
  arma::mat S_save(L, m, fill::zeros);
  arma::mat Cell_save(K, total_cell_num, fill::zeros);
  arma::mat mu_save(G, K, fill::zeros);
  arma::mat beta_save(G, L, fill::zeros);
  arma::mat Pi_save(K, L, fill::zeros);
  
  arma::vec sigma_sq_save(G, fill::zeros);
  arma::vec lam0_save(G, fill::zeros);
  arma::vec lam1_save(G, fill::zeros);
  
  // double eta_mu_save = 0;
  // double tau_sq_mu_save = 0;
  // double eta_beta_save = 0;
  // double tau_sq_beta_save = 0;
  double alpha_save = 0;
  arma::vec phi_save(L, fill::zeros);
  
  if (iter_save) {
    arma::cube mu_iter(G, K, n_save);
    arma::cube beta_iter(G, L, n_save);
    arma::mat sigma_sq_iter(G, n_save);
    arma::mat lam0_iter(G, n_save);
    arma::mat lam1_iter(G, n_save);
    
    // arma::vec eta_mu_iter(n_save);
    // arma::vec tau_sq_mu_iter(n_save);
    // arma::vec eta_beta_iter(n_save);
    // arma::vec tau_sq_beta_iter(n_save);
    arma::vec alpha_iter(n_save);
    arma::mat phi_iter(L, n_save);
    // arma::mat xi_iter(K, n_save);
    
    if (class_print) {
      Environment base("package:base"); 
      Function table = base["table"];
      
      SEXP S_table = table(Rcpp::_["..."] = S_t);
      irowvec S_out = Rcpp::as<arma::irowvec>(S_table);
      SEXP Cell_table = table(Rcpp::_["..."] = clas_cell_t);
      irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
      cout<<"Iteration "<< 0 <<endl;
      cout<<"Subject"<<endl; 
      cout<<S_out<<endl; 
      cout<<"Cell"<<endl; 
      cout<<Cell_out<<endl; 
      
      for (int t_iter = 0; t_iter < n_iter; t_iter++) {
        update_theta(theta_t, y_t, mu_t, beta_t, clas_cell_t, sigma_sq_t, scale_factor_est, S_vec_t, G, total_cell_num, tau_theta, n_threads);
        
        update_y(y_t, theta_t, ind_zero, scale_factor_est, lambda0_t, lambda1_t, G, total_cell_num, radius, n_threads);
        
        update_lambda(lambda0_t, lambda1_t, y_t, ind_zero, G, lam0_0, lam1_0, Sigma2_lam0, Sigma2_lam1, sd_lambda0, sd_lambda1, n_threads);
        
        update_mu(mu_t, theta_t, beta_t, sigma_sq_t, clas_cell_t, S_vec_t, K, G, eta_mu, tau_sq_mu);
        update_beta(beta_t, theta_t, mu_t, sigma_sq_t, clas_cell_t, S_vec_t, L, G, eta_beta, tau_sq_beta);
        mu_t.each_col() += beta_t.col(S_t(0));
        beta_t.each_col() -= beta_t.col(S_t(0));
        
        update_mu_prior(eta_mu, tau_sq_mu, mu_t, K, G);
        
        update_beta_prior(eta_beta, tau_sq_beta, beta_t, L, G, S_t(0));
        
        update_sigma_sq(sigma_sq_t, theta_t, mu_t, beta_t, clas_cell_t, S_vec_t, G, total_cell_num, n_threads);
        
        update_clas(S_t, S_vec_t, clas_cell_t, theta_t, mu_t, beta_t, Pi_t, ind_stat_cell, sigma_sq_t, phi_t, n_cell, G, total_cell_num, L, K, m, n_threads);
        
        update_phi(phi_t, S_t, nu_t, L);
        
        update_Pi(Pi_t, clas_cell_t, S_vec_t, xi_vec_t, gam_t, L, K);
        
        update_xi_alpha(xi_vec_t, xi_prime_t, alpha_t, Pi_t, gam_t, K, L, a_xi, b_xi);
        
        SEXP S_table = table(Rcpp::_["..."] = S_t);
        irowvec S_out = Rcpp::as<arma::irowvec>(S_table);
        SEXP Cell_table = table(Rcpp::_["..."] = clas_cell_t);
        irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
        cout<<"Iteration "<< t_iter+1 <<endl;
        cout<<"Subject"<<endl; 
        cout<<S_out<<endl; 
        cout<<"Cell"<<endl; 
        cout<<Cell_out<<endl; 
        
        if (t_iter >= save_start) {
          int save_i = t_iter - save_start;
          for (int S_i = 0; S_i < m; S_i++) {
            S_save(S_t(S_i),S_i) += 1;
          }
          for (int C_i = 0; C_i < total_cell_num; C_i++) {
            Cell_save(clas_cell_t(C_i),C_i) += 1;
          }
          mu_save += mu_t;
          beta_save += beta_t;
          Pi_save += Pi_t;
          
          sigma_sq_save += sigma_sq_t;
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          
          // eta_mu_save += eta_mu;
          // tau_sq_mu_save += tau_sq_mu;
          // eta_beta_save += eta_beta;
          // tau_sq_beta_save += tau_sq_beta;
          alpha_save += alpha_t;
          phi_save += phi_t;
          
          mu_iter.slice(save_i) = mu_t;
          beta_iter.slice(save_i) = beta_t;
          sigma_sq_iter.col(save_i) = sigma_sq_t;
          lam0_iter.col(save_i) = lambda0_t;
          lam1_iter.col(save_i) = lambda1_t;
          
          // eta_mu_iter(save_i) = eta_mu;
          // tau_sq_mu_iter(save_i) = tau_sq_mu;
          // eta_beta_iter(save_i) = eta_beta;
          // tau_sq_beta_iter(save_i) = tau_sq_beta;
          alpha_iter(save_i) = alpha_t;
          
          phi_iter.col(save_i) = phi_t;
          // xi_iter.col(save_i) = xi_vec_t;
        }
      }
      
      mu_save /= n_save;
      beta_save /= n_save;
      Pi_save /= n_save;
      
      sigma_sq_save /= n_save;
      lam0_save /= n_save;
      lam1_save /= n_save;
      
      // eta_mu_save /= n_save;
      // tau_sq_mu_save /= n_save;
      // eta_beta_save /= n_save;
      // tau_sq_beta_save /= n_save;
      alpha_save /= n_save;
      phi_save /= n_save;
      
      arma::vec S_est(m);
      for (int S_i = 0; S_i < m; S_i++) {
        S_est(S_i) = S_save.col(S_i).index_max() + 1;
      }
      
      arma::vec Cell_est(total_cell_num);
      for (int C_i = 0; C_i < total_cell_num; C_i++) {
        Cell_est(C_i) = Cell_save.col(C_i).index_max() + 1;
      }
      
      
      return List::create(Named("subject_subgroup_label")=S_est, Named("cell_type_label")=Cell_est, Named("cell_type_effects")=mu_save, Named("subject_subgroup_effects")=beta_save, Named("cell_type_prop")=Pi_save,
        Named("cell_type_effects_post")=mu_iter, Named("subject_subgroup_effects_post")=beta_iter, Named("sigma_sq_post")=sigma_sq_iter, Named("lam0_post")=lam0_iter, 
        Named("lam1_post")=lam1_iter, Named("alpha_post")=alpha_iter, Named("subject_subgroup_prop_post")=phi_iter,
              // Named("eta_mu_iter")=eta_mu_iter, Named("tau_sq_mu_iter")=tau_sq_mu_iter, Named("eta_beta_iter")=eta_beta_iter, Named("tau_sq_beta_iter")=tau_sq_beta_iter,
              Named("sigma_sq")=sigma_sq_save, Named("lam0")=lam0_save, Named("lam1")=lam1_save, 
              // Named("eta_mu")=eta_mu_save, Named("tau_sq_mu")=tau_sq_mu_save, Named("eta_beta")=eta_beta_save, Named("tau_sq_beta")=tau_sq_beta_save, 
              Named("alpha")=alpha_save, Named("subject_subgroup_prop")=phi_save);
    } else {
      
      for (int t_iter = 0; t_iter < n_iter; t_iter++) {
        update_theta(theta_t, y_t, mu_t, beta_t, clas_cell_t, sigma_sq_t, scale_factor_est, S_vec_t, G, total_cell_num, tau_theta, n_threads);
        
        update_y(y_t, theta_t, ind_zero, scale_factor_est, lambda0_t, lambda1_t, G, total_cell_num, radius, n_threads);
        
        update_lambda(lambda0_t, lambda1_t, y_t, ind_zero, G, lam0_0, lam1_0, Sigma2_lam0, Sigma2_lam1, sd_lambda0, sd_lambda1, n_threads);
        
        update_mu(mu_t, theta_t, beta_t, sigma_sq_t, clas_cell_t, S_vec_t, K, G, eta_mu, tau_sq_mu);
        update_beta(beta_t, theta_t, mu_t, sigma_sq_t, clas_cell_t, S_vec_t, L, G, eta_beta, tau_sq_beta);
        mu_t.each_col() += beta_t.col(S_t(0));
        beta_t.each_col() -= beta_t.col(S_t(0));
        
        update_mu_prior(eta_mu, tau_sq_mu, mu_t, K, G);
        
        update_beta_prior(eta_beta, tau_sq_beta, beta_t, L, G, S_t(0));
        
        update_sigma_sq(sigma_sq_t, theta_t, mu_t, beta_t, clas_cell_t, S_vec_t, G, total_cell_num, n_threads);
        
        update_clas(S_t, S_vec_t, clas_cell_t, theta_t, mu_t, beta_t, Pi_t, ind_stat_cell, sigma_sq_t, phi_t, n_cell, G, total_cell_num, L, K, m, n_threads);
        
        update_phi(phi_t, S_t, nu_t, L);
        
        update_Pi(Pi_t, clas_cell_t, S_vec_t, xi_vec_t, gam_t, L, K);
        
        update_xi_alpha(xi_vec_t, xi_prime_t, alpha_t, Pi_t, gam_t, K, L, a_xi, b_xi);
        
        if ((t_iter+1) % iter_print == 0) {
          // iter_out("step.txt", t_iter);
          cout<<"Iteration "<< t_iter+1 <<endl;
        }
        
        if (t_iter >= save_start) {
          int save_i = t_iter - save_start;
          for (int S_i = 0; S_i < m; S_i++) {
            S_save(S_t(S_i),S_i) += 1;
          }
          for (int C_i = 0; C_i < total_cell_num; C_i++) {
            Cell_save(clas_cell_t(C_i),C_i) += 1;
          }
          mu_save += mu_t;
          beta_save += beta_t;
          Pi_save += Pi_t;
          
          sigma_sq_save += sigma_sq_t;
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          
          // eta_mu_save += eta_mu;
          // tau_sq_mu_save += tau_sq_mu;
          // eta_beta_save += eta_beta;
          // tau_sq_beta_save += tau_sq_beta;
          alpha_save += alpha_t;
          phi_save += phi_t;
          
          mu_iter.slice(save_i) = mu_t;
          beta_iter.slice(save_i) = beta_t;
          sigma_sq_iter.col(save_i) = sigma_sq_t;
          lam0_iter.col(save_i) = lambda0_t;
          lam1_iter.col(save_i) = lambda1_t;
          
          // eta_mu_iter(save_i) = eta_mu;
          // tau_sq_mu_iter(save_i) = tau_sq_mu;
          // eta_beta_iter(save_i) = eta_beta;
          // tau_sq_beta_iter(save_i) = tau_sq_beta;
          alpha_iter(save_i) = alpha_t;
          
          phi_iter.col(save_i) = phi_t;
          // xi_iter.col(save_i) = xi_vec_t;
        }
      }
      
      mu_save /= n_save;
      beta_save /= n_save;
      Pi_save /= n_save;
      
      sigma_sq_save /= n_save;
      lam0_save /= n_save;
      lam1_save /= n_save;
      
      // eta_mu_save /= n_save;
      // tau_sq_mu_save /= n_save;
      // eta_beta_save /= n_save;
      // tau_sq_beta_save /= n_save;
      alpha_save /= n_save;
      phi_save /= n_save;
      
      arma::vec S_est(m);
      for (int S_i = 0; S_i < m; S_i++) {
        S_est(S_i) = S_save.col(S_i).index_max() + 1;
      }
      
      arma::vec Cell_est(total_cell_num);
      for (int C_i = 0; C_i < total_cell_num; C_i++) {
        Cell_est(C_i) = Cell_save.col(C_i).index_max() + 1;
      }
      
      
      return List::create(Named("subject_subgroup_label")=S_est, Named("cell_type_label")=Cell_est, Named("cell_type_effects")=mu_save, Named("subject_subgroup_effects")=beta_save, Named("cell_type_prop")=Pi_save,
        Named("cell_type_effects_post")=mu_iter, Named("subject_subgroup_effects_post")=beta_iter, Named("sigma_sq_post")=sigma_sq_iter, Named("lam0_post")=lam0_iter, 
        Named("lam1_post")=lam1_iter, Named("alpha_post")=alpha_iter, Named("subject_subgroup_prop_post")=phi_iter,
              // Named("eta_mu_iter")=eta_mu_iter, Named("tau_sq_mu_iter")=tau_sq_mu_iter, Named("eta_beta_iter")=eta_beta_iter, Named("tau_sq_beta_iter")=tau_sq_beta_iter,
              Named("sigma_sq")=sigma_sq_save, Named("lam0")=lam0_save, Named("lam1")=lam1_save, 
              // Named("eta_mu")=eta_mu_save, Named("tau_sq_mu")=tau_sq_mu_save, Named("eta_beta")=eta_beta_save, Named("tau_sq_beta")=tau_sq_beta_save, 
              Named("alpha")=alpha_save, Named("subject_subgroup_prop")=phi_save);
    }
  } else {
    
    if (class_print) {
      Environment base("package:base"); 
      Function table = base["table"];
      
      SEXP S_table = table(Rcpp::_["..."] = S_t);
      irowvec S_out = Rcpp::as<arma::irowvec>(S_table);
      SEXP Cell_table = table(Rcpp::_["..."] = clas_cell_t);
      irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
      cout<<"Iteration "<< 0 <<endl;
      cout<<"Subject"<<endl; 
      cout<<S_out<<endl; 
      cout<<"Cell"<<endl; 
      cout<<Cell_out<<endl;
      
      for (int t_iter = 0; t_iter < n_iter; t_iter++) {
        update_theta(theta_t, y_t, mu_t, beta_t, clas_cell_t, sigma_sq_t, scale_factor_est, S_vec_t, G, total_cell_num, tau_theta, n_threads);
        
        update_y(y_t, theta_t, ind_zero, scale_factor_est, lambda0_t, lambda1_t, G, total_cell_num, radius, n_threads);
        
        update_lambda(lambda0_t, lambda1_t, y_t, ind_zero, G, lam0_0, lam1_0, Sigma2_lam0, Sigma2_lam1, sd_lambda0, sd_lambda1, n_threads);
        
        update_mu(mu_t, theta_t, beta_t, sigma_sq_t, clas_cell_t, S_vec_t, K, G, eta_mu, tau_sq_mu);
        update_beta(beta_t, theta_t, mu_t, sigma_sq_t, clas_cell_t, S_vec_t, L, G, eta_beta, tau_sq_beta);
        mu_t.each_col() += beta_t.col(S_t(0));
        beta_t.each_col() -= beta_t.col(S_t(0));
        
        update_mu_prior(eta_mu, tau_sq_mu, mu_t, K, G);
        
        update_beta_prior(eta_beta, tau_sq_beta, beta_t, L, G, S_t(0));
        
        update_sigma_sq(sigma_sq_t, theta_t, mu_t, beta_t, clas_cell_t, S_vec_t, G, total_cell_num, n_threads);
        
        update_clas(S_t, S_vec_t, clas_cell_t, theta_t, mu_t, beta_t, Pi_t, ind_stat_cell, sigma_sq_t, phi_t, n_cell, G, total_cell_num, L, K, m, n_threads);
        
        update_phi(phi_t, S_t, nu_t, L);
        
        update_Pi(Pi_t, clas_cell_t, S_vec_t, xi_vec_t, gam_t, L, K);
        
        update_xi_alpha(xi_vec_t, xi_prime_t, alpha_t, Pi_t, gam_t, K, L, a_xi, b_xi);
        
        SEXP S_table = table(Rcpp::_["..."] = S_t);
        irowvec S_out = Rcpp::as<arma::irowvec>(S_table);
        SEXP Cell_table = table(Rcpp::_["..."] = clas_cell_t);
        irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
        cout<<"Iteration "<< t_iter+1 <<endl;
        cout<<"Subject"<<endl; 
        cout<<S_out<<endl; 
        cout<<"Cell"<<endl; 
        cout<<Cell_out<<endl; 
        
        if (t_iter >= save_start) {
          for (int S_i = 0; S_i < m; S_i++) {
            S_save(S_t(S_i),S_i) += 1;
          }
          for (int C_i = 0; C_i < total_cell_num; C_i++) {
            Cell_save(clas_cell_t(C_i),C_i) += 1;
          }
          mu_save += mu_t;
          beta_save += beta_t;
          Pi_save += Pi_t;
          
          sigma_sq_save += sigma_sq_t;
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          
          // eta_mu_save += eta_mu;
          // tau_sq_mu_save += tau_sq_mu;
          // eta_beta_save += eta_beta;
          // tau_sq_beta_save += tau_sq_beta;
          alpha_save += alpha_t;
          phi_save += phi_t;
        }
      }
      
      mu_save /= n_save;
      beta_save /= n_save;
      Pi_save /= n_save;
      
      sigma_sq_save /= n_save;
      lam0_save /= n_save;
      lam1_save /= n_save;
      
      // eta_mu_save /= n_save;
      // tau_sq_mu_save /= n_save;
      // eta_beta_save /= n_save;
      // tau_sq_beta_save /= n_save;
      alpha_save /= n_save;
      phi_save /= n_save;
      
      arma::vec S_est(m);
      for (int S_i = 0; S_i < m; S_i++) {
        S_est(S_i) = S_save.col(S_i).index_max() + 1;
      }
      
      arma::vec Cell_est(total_cell_num);
      for (int C_i = 0; C_i < total_cell_num; C_i++) {
        Cell_est(C_i) = Cell_save.col(C_i).index_max() + 1;
      }
      return List::create(Named("subject_subgroup_label")=S_est, Named("cell_type_label")=Cell_est, Named("cell_type_effects")=mu_save, Named("subject_subgroup_effects")=beta_save, Named("cell_type_prop")=Pi_save,
        Named("sigma_sq")=sigma_sq_save, Named("lam0")=lam0_save, Named("lam1")=lam1_save, 
        // Named("eta_mu")=eta_mu_save, Named("tau_sq_mu")=tau_sq_mu_save, Named("eta_beta")=eta_beta_save, Named("tau_sq_beta")=tau_sq_beta_save, 
        Named("alpha")=alpha_save, Named("subject_subgroup_prop")=phi_save);
    } else {
      
      for (int t_iter = 0; t_iter < n_iter; t_iter++) {
        update_theta(theta_t, y_t, mu_t, beta_t, clas_cell_t, sigma_sq_t, scale_factor_est, S_vec_t, G, total_cell_num, tau_theta, n_threads);
        
        update_y(y_t, theta_t, ind_zero, scale_factor_est, lambda0_t, lambda1_t, G, total_cell_num, radius, n_threads);
        
        update_lambda(lambda0_t, lambda1_t, y_t, ind_zero, G, lam0_0, lam1_0, Sigma2_lam0, Sigma2_lam1, sd_lambda0, sd_lambda1, n_threads);
        
        update_mu(mu_t, theta_t, beta_t, sigma_sq_t, clas_cell_t, S_vec_t, K, G, eta_mu, tau_sq_mu);
        update_beta(beta_t, theta_t, mu_t, sigma_sq_t, clas_cell_t, S_vec_t, L, G, eta_beta, tau_sq_beta);
        mu_t.each_col() += beta_t.col(S_t(0));
        beta_t.each_col() -= beta_t.col(S_t(0));
        
        update_mu_prior(eta_mu, tau_sq_mu, mu_t, K, G);
        
        update_beta_prior(eta_beta, tau_sq_beta, beta_t, L, G, S_t(0));
        
        update_sigma_sq(sigma_sq_t, theta_t, mu_t, beta_t, clas_cell_t, S_vec_t, G, total_cell_num, n_threads);
        
        update_clas(S_t, S_vec_t, clas_cell_t, theta_t, mu_t, beta_t, Pi_t, ind_stat_cell, sigma_sq_t, phi_t, n_cell, G, total_cell_num, L, K, m, n_threads);
        
        update_phi(phi_t, S_t, nu_t, L);
        
        update_Pi(Pi_t, clas_cell_t, S_vec_t, xi_vec_t, gam_t, L, K);
        
        update_xi_alpha(xi_vec_t, xi_prime_t, alpha_t, Pi_t, gam_t, K, L, a_xi, b_xi);
        
        if ((t_iter+1) % iter_print == 0) {
          // iter_out("step.txt", t_iter);
          cout<<"Iteration "<< t_iter+1 <<endl;
        }
        
        if (t_iter >= save_start) {
          for (int S_i = 0; S_i < m; S_i++) {
            S_save(S_t(S_i),S_i) += 1;
          }
          for (int C_i = 0; C_i < total_cell_num; C_i++) {
            Cell_save(clas_cell_t(C_i),C_i) += 1;
          }
          mu_save += mu_t;
          beta_save += beta_t;
          Pi_save += Pi_t;
          
          sigma_sq_save += sigma_sq_t;
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          
          // eta_mu_save += eta_mu;
          // tau_sq_mu_save += tau_sq_mu;
          // eta_beta_save += eta_beta;
          // tau_sq_beta_save += tau_sq_beta;
          alpha_save += alpha_t;
          phi_save += phi_t;
        }
      }
      
      mu_save /= n_save;
      beta_save /= n_save;
      Pi_save /= n_save;
      
      sigma_sq_save /= n_save;
      lam0_save /= n_save;
      lam1_save /= n_save;
      
      // eta_mu_save /= n_save;
      // tau_sq_mu_save /= n_save;
      // eta_beta_save /= n_save;
      // tau_sq_beta_save /= n_save;
      alpha_save /= n_save;
      phi_save /= n_save;
      
      arma::vec S_est(m);
      for (int S_i = 0; S_i < m; S_i++) {
        S_est(S_i) = S_save.col(S_i).index_max() + 1;
      }
      
      arma::vec Cell_est(total_cell_num);
      for (int C_i = 0; C_i < total_cell_num; C_i++) {
        Cell_est(C_i) = Cell_save.col(C_i).index_max() + 1;
      }
      return List::create(Named("subject_subgroup_label")=S_est, Named("cell_type_label")=Cell_est, Named("cell_type_effects")=mu_save, Named("subject_subgroup_effects")=beta_save, Named("cell_type_prop")=Pi_save,
        Named("sigma_sq")=sigma_sq_save, Named("lam0")=lam0_save, Named("lam1")=lam1_save, 
        // Named("eta_mu")=eta_mu_save, Named("tau_sq_mu")=tau_sq_mu_save, Named("eta_beta")=eta_beta_save, Named("tau_sq_beta")=tau_sq_beta_save, 
        Named("alpha")=alpha_save, Named("subject_subgroup_prop")=phi_save);
    }
  }
}



// [[Rcpp::export]]
arma::mat update_mu_R(arma::mat theta_t, arma::mat beta_t, arma::vec sigma_sq_t, arma::vec clas_cell_t, arma::uvec S_vec_t, 
                      const int K, const int G, double eta_mu, double tau_sq_mu) {
  arma::mat mu_updated(G, K, fill::zeros);
  for (int k = 0; k < K; k++) {
    arma::vec tmp1(G, fill::zeros);
    double tmp2 = 0;
    
    arma::uvec ind_k = arma::find(clas_cell_t == (k+1));
    if (ind_k.n_elem > 0) {
      arma::uvec S_k = S_vec_t(ind_k);
      tmp1 = arma::sum(theta_t.rows(ind_k).t() - beta_t.cols(S_k-1), 1);
      tmp2 = ind_k.n_elem;
    }
    
    arma::vec avg = (tmp1/sigma_sq_t + eta_mu/tau_sq_mu)/(tmp2/sigma_sq_t + 1.0/tau_sq_mu);
    arma::vec sd = sqrt(1.0/(tmp2/sigma_sq_t + 1.0/tau_sq_mu));
    for (int i = 0; i < G; i++){
      mu_updated(i,k) = R::rnorm(avg(i), sd(i));
    }
  }
  return mu_updated;
}


// [[Rcpp::export]]
arma::mat update_Pi_R(arma::uvec clas_cell_t, arma::vec S_vec_t, arma::vec xi_vec_t, double gam, const int L, const int K) {
  arma::vec xi_gamma = xi_vec_t * gam;
  arma::mat tmp(K, L, fill::none);
  for (int ell = 0; ell < L; ell++) {
    arma::vec tmp0(K, fill::zeros);
    arma::uvec ind_ell = arma::find(S_vec_t == (ell+1));
    if (ind_ell.n_elem > 0) {
      arma::uvec clas_cell_ell = clas_cell_t(ind_ell);
      for (int k = 0; k < K; k++) {
        arma::uvec ind_j = arma::find(clas_cell_ell == (k+1));
        tmp0(k) = ind_j.n_elem;
      }
    }
    tmp.col(ell) = rDirichlet(tmp0+xi_gamma);
  }
  return tmp;
}
