#' Simultaneous subject and cell clustering for single cell expression count data
#'
#' The function SCSC implements a nonparametric Bayesian model for simultaneous subject subgroup discovery and cell type detection based on the scRNA-seq data from multiple subjects. It does not need to prespecify the exact subject subgroup number or cell type number but only their upper bounds, and automatically induces subject subgroup structures and matches cell types across subjects. SCSC is directly applied to the scRNA-seq raw count data because it deliberately considers the data's dropouts, library sizes, and over-dispersion.  In SCSC, a blocked Gibbs sampler is carried out for Bayesian posterior inference.
#' 
#' @param count_data_matr the expression count data matrix, where rows represent genes and columns represent cells. Noteworthy, the columns need to be ordered based on each cell’s subject information. Specifically, cells of subject 1 correspond to columns 1 to vec_ncell_subj[1], cells of subject 2 match columns (vec_ncell_subj[1]+1):(vec_ncell_subj[1]+vec_ncell_subj[2]) , and so on so forth. The vector vec_ncell_subj is specified next.
#' @param vec_ncell_subj an integer vector, in which element i is the total cell number of subject i.
#' @param subgroup_upb an integer, denoting an upper bound of the subject subgroup number.
#' @param celltype_upb an integer, denoting an upper bound of the cell type number.
#' @param warm_cluster_label_init the initialization of cluster labels is random or based on k-means. The default is FALSE, corresponding to the random initialization.
#' @param con_para_nu the concentration parameter \eqn{\nu} in NDP used to generate subject subgroup proportions. The default is 0.1.
#' @param con_para_gamma the concentration parameter \eqn{\gamma} in NDP used to generate cell type proportions. The default is 0.1.
#' @param num_iterations the number of Gibbs sampler iterations. The default is 10000.
#' @param num_burnin the number of iterations in burn-in, after which the posterior samples are used to estimate the unknown parameters. The default is the first half of total iterations.
#' @param collect_post_sample collect the posterior samples or not. If users are only interested in the estimates, set it as FALSE to save the memory. If users would like to use posterior samples o construct credible intervals or for other uses, set it as TRUE. The default is FALSE.
#' @param hyperparameters a vector, which indicates 10 hyper-parameters in the priors or proposal distributions. The first element is the standard deviation of the normal proposal distribution for \eqn{\theta} in the MH step. The second element is an integer representing the radius of the discrete uniform proposal distribution for latent count in the MH step. The third element represents the standard deviation of the normal proposal distribution for the coefficient \eqn{\lambda_0}, while the fourth element is the standard deviation of the normal proposal distribution for \eqn{\lambda_1}. The fifth and sixth elements are the shape parameters of the beta proposal distribution for \eqn{\xi'}, respectively. The seventh and eighth elements correspond to the means of the normal priors for \eqn{\lambda_0} and \eqn{\lambda_1}. The remaining two elements are, respectively, the variances of the normal priors for \eqn{\lambda_0} and \eqn{\lambda_1}. The default is 1, 5, 0.15, 0.15, 1, 1, 2, -2, 0.01, 0.01.
#' @param scaling_factor_est estimate the scaling factors for cells or not. The default is FALSE, which fixes all scaling factors at one.
#' @param print_label whether or not to print summarized cell type label and subject subgroup label information after each iteration. The default is FALSE.
#' @param seed set seed to reproduce results. The default is 1.
#' @param num_threads the number of threads generated in the parallel computing. The default is 1.
#' @param print_per_iteration how many iterations to print the iteration information when print_label is FALSE. The default is 1000.
#' 
#' @return SCSC returns an R list including the following information.
#' \item{subject_subgroup_label}{a vector, indicating the estimated subgroup labels for each subject.}
#' \item{cell_type_label}{a vector, indicating the estimated cell types for each cell.}
#' \item{subject_subgroup_effects}{a matrix of subjectsubgroup effects, in which rows are genes and columns correspond to subject subgroups. Note that the first column is a zero vector because the first subgroup is treated as the reference subgroup, where there is no subject subgroup effect.}
#' \item{cell_type_effects}{a matrix of cell type effects, in which rows are genes and columns correspond to cell types.}
#' \item{cell_type_prop}{a matrix of cell type proportions for each subject subgroup, in which rows are cell types and columns are subject subgroups. Specifically, cell_type_prop[k, \eqn{\ell}] is the cell type k proportion of subject subgroup \eqn{\ell}.}
#' \item{sigma_sq}{a vector, the estimated expression variances for each gene.}
#' \item{lam0}{a vector, the estimated lambda0 for each gene.}
#' \item{lam1}{a vector, the estimated lambda1 for each gene.}
#' \item{alpha}{a scalar, denoting the estimated concentration parameter \eqn{\alpha} in DP at the top level.}
#' \item{subject_subgroup_prop}{a vector denoting the estimated subject subgroup proportions.}
#' \item{subject_subgroup_effects_post}{collected posterior samples of subject_subgroup_effects when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{cell_type_effects_post}{collected posterior samples of cell_type_effects when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{sigma_sq_post}{collected posterior samples of sigma_sq when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{lam0_post}{collected posterior samples of lam0 when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{lam1_post}{collected posterior samples of lam1 when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{alpha_post}{collected posterior samples of alpha when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{subject_subgroup_prop_post}{collected posterior samples of subject_subgroup_prop when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' 
#' @examples 
#' library(SCSC)
#' 
#' #import example data
#' data(example_data)
#' 
#' #gene number
#' nrow(count_data_matr)
#' 
#' #cell number
#' ncol(count_data_matr)
#' 
#' #subject number
#' length(vec_ncell_subj)
#' 
#' #run SCSC
#' t1 <- Sys.time()
#' Result <- SCSC(count_data_matr, vec_ncell_subj, celltype_upb = 10, subgroup_upb = 10,
#'       seed = 1, num_threads = 10, num_iterations = 1000, print_label = TRUE)
#' t2 <- Sys.time()
#' 
#' #time cost
#' print(t2 - t1)
#' 
#' #Compared with true subject subgroup labels
#' table(Result$subject_subgroup_label, subject_subgroup_label_truth)
#' 
#' #Compared with true cell type labels
#' cell_table <- table(Result$cell_type_label, cell_type_label_truth)
#' cell_table
#' 
#' #The following shows the summary of the absolute errors across genes within each subject subgroup
#' summary(abs(Result$subject_subgroup_effects - subject_subgroup_effects_truth))
#' 
#' #The following shows the summary of the absolute errors across genes within each cell type
#' type_name <- rownames(which(cell_table > 0,TRUE))
#' cell_unique <- unique(Result$cell_type_label)
#' summary(abs(Result$cell_type_effects[,c(which(type_name[1]==cell_unique)
#'             ,which(type_name[2]==cell_unique), which(type_name[3]==cell_unique))]
#'              - cell_type_effects_truth))
#' 
#' @references Qiuyu Wu, and Xiangyu Luo. "Nonparametric Bayesian Two-Level Clustering for Subject-Level Single-Cell Expression Data." Statistica Sinica. DOI: 10.5705/ss.202020.0337
#' @export
SCSC <- function(count_data_matr, vec_ncell_subj, celltype_upb, subgroup_upb, warm_cluster_label_init = FALSE, con_para_nu = 0.1, con_para_gamma = 0.1, num_iterations = 10000, num_burnin = floor(num_iterations/2),
                 collect_post_sample = FALSE, hyperparameters = c(1, 5, 0.15, 0.15, 1, 1, 2, -2, 0.01, 0.01), scaling_factor_est = FALSE, print_label = FALSE,
                 seed = 1, num_threads = 1, print_per_iteration = 1000) {
  
  count_data_matr <- t(as.matrix(count_data_matr))
  G <- dim(count_data_matr)[2]
  num_subjects <- length(vec_ncell_subj)
  total_cell_num <- sum(vec_ncell_subj)
  ind_stat_cell <- c(1, 1+cumsum(vec_ncell_subj))
  ind_stat_cell <- ind_stat_cell[1:num_subjects]
  
  
  #estimate scaling factor
  if (scaling_factor_est) {
    scale_factor_est <- NULL
    for (j in 1:num_subjects){
      tmp <- rowSums(count_data_matr[ind_stat_cell[j]:(ind_stat_cell[j]+vec_ncell_subj[j]-1), ])
      tmp <- tmp / median(tmp)
      scale_factor_est <- c(scale_factor_est, tmp)
    }
  } else {
    scale_factor_est <- rep(1, total_cell_num)
  }
  #zero index
  ind_zero <- (count_data_matr == 0)
  
  #set seed to reproduce results
  set.seed(seed)
  
  
  #initialize the partially latent matrix y_t
  y_t <-  count_data_matr
  for (g in 1:G) {
    y_t[ind_zero[, g], g] <- quantile(count_data_matr[!ind_zero[, g], g], probs = 0.01)
  }
  
  #initialize theta_t
  theta_t <- log((y_t/scale_factor_est)+1)
  
  #randomly initialize the subject subgroup indicators
  # L_rand <- sample(1:subgroup_upb, subgroup_upb)
  # S_t <- rep(L_rand,ceiling(num_subjects/subgroup_upb))[1:num_subjects]
  if (warm_cluster_label_init == FALSE) {
    S_t <- sample(1:subgroup_upb, num_subjects, replace = TRUE)
    clas_cell_t <- sample(1:celltype_upb, total_cell_num, replace = TRUE)
  } else {
    subjectmatrix <- NULL
    for (j in 1:num_subjects) {
      subjectmatrix <- cbind(subjectmatrix,apply(log2(count_data_matr[ind_stat_cell[j]:(ind_stat_cell[j]+vec_ncell_subj[j]-1), ]+1),2,mean))
    }
    S_t <- kmeans(subjectmatrix,subgroup_upb)$cluster
    clas_cell_t <- kmeans(log2(count_data_matr+1),celltype_upb)$cluster
  }
  
  S_vec_t <- NULL
  for (i in 1:num_subjects) {
    S_vec_t <- c(S_vec_t, rep(S_t[i], vec_ncell_subj[i]))
  }
  
  #initialize alpha_vec_t
  alpha_t <- 0.5
  
  #initialize xi_prime_t and xi_vec_t
  xi_prime_t <- c(rbeta(celltype_upb-1, 1, alpha_t), 1)
  xi_vec_t <- xi_prime_t[1]
  for (celltype_upb in 2:celltype_upb) {
    xi_vec_t <- c(xi_vec_t, (1-sum(xi_vec_t))*xi_prime_t[celltype_upb])
  }
  while (sum(xi_vec_t == 0) > 0) {
    xi_prime_t <- c(rbeta(celltype_upb-1, 1, alpha_t), 1)
    xi_vec_t <- xi_prime_t[1]
    for (celltype_upb in 2:celltype_upb) {
      xi_vec_t <- c(xi_vec_t, (1-sum(xi_vec_t))*xi_prime_t[celltype_upb])
    }
  }
  
  #initialize nu_t
  con_para_nu <- 0.1
  
  #initialize gam_t
  con_para_gamma <- 0.1
  
  #initialize phi_t
  phi_t <- NULL
  for(ell in 1:subgroup_upb){
    phi_t <- c(phi_t, sum(S_t==ell)/num_subjects)
  }
  
  #initialize Pi_t
  Pi_t <- update_Pi_R(clas_cell_t, S_vec_t, xi_vec_t, con_para_gamma, subgroup_upb, celltype_upb)
  
  #initialize beta_t
  beta_t <- matrix(0, G, subgroup_upb)
  beta_t[,2:subgroup_upb] <- matrix(rnorm(G*(subgroup_upb-1),0,2),G,subgroup_upb-1)
  
  #initialize sigma_sq_t
  sigma_sq_t <- rep(1, G)
  
  #initialize parameters in priors
  eta_mu <- 0
  tau_sq_mu <- 2
  eta_beta <- 0
  tau_sq_beta<- 2
  
  #initialize mu_t
  mu_t <- update_mu_R(theta_t, beta_t, sigma_sq_t, clas_cell_t, S_vec_t, celltype_upb, G, eta_mu, tau_sq_mu)
  
  #initialize lambda_t
  lambda0_t <- rnorm(G, mean=hyperparameters[7], sd=hyperparameters[9])
  lambda1_t <- rnorm(G, mean=hyperparameters[8], sd=hyperparameters[10])
  
  ###############################################################
  ################### The Blocked Gibbs Sampler #################
  ###############################################################
  num_saved <- num_iterations - num_burnin
  
  Result <- MCMC_full(num_iterations, num_saved, theta_t, y_t, ind_zero, beta_t, mu_t, Pi_t, 
                      ind_stat_cell, phi_t, vec_ncell_subj, sigma_sq_t, clas_cell_t, lambda0_t, lambda1_t,
                      S_t, S_vec_t, xi_vec_t, xi_prime_t, scale_factor_est,
                      alpha_t, con_para_nu, con_para_gamma, eta_mu, tau_sq_mu, eta_beta, tau_sq_beta,
                      G, total_cell_num, subgroup_upb, celltype_upb, num_subjects,
                      hyperparameters[1], hyperparameters[2], hyperparameters[7], hyperparameters[8], hyperparameters[9], 
                      hyperparameters[10], hyperparameters[3], hyperparameters[4], hyperparameters[5], hyperparameters[6],
                      collect_post_sample, num_threads, print_per_iteration, print_label)
  
  Result$cell_type_effects <- Result$cell_type_effects[, unique(Result$cell_type_label)] 
  Result$subject_subgroup_effects <- Result$subject_subgroup_effects[, unique(Result$subject_subgroup_label)]
  
  return(Result)
}