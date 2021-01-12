## SCSC: simultaneous Subject and Cell clustering for Single Cell expression count data
This R package aims at the implementation of a nonparametric Bayesian model named SCSC for simultaneous subject subgroup discovery and cell type detection based on the scRNA-seq data from multiple subjects. SCSC does not need to prespecify the exact subject subgroup number or cell type number but only their upper bounds, and automatically induces subject subgroup structures and matches cell types across subjects. SCSC is directly applied to the scRNA-seq raw count data owing to its consideration of the data's dropouts, library sizes and over-dispersion. In this package, a blocked Gibbs sampler is carried out for Bayesian posterior inference of SCSC.

For technical details, please refer to the arxiv paper: Qiuyu Wu, and Xiangyu Luo. "Nonparametric Bayesian Two-Level Clustering for Subject-Level Single-Cell Expression Data." arXiv:1912.08050  <https://arxiv.org/abs/1912.08050>. 

The code that can reproduce results in the paper can be downloaded through https://drive.google.com/file/d/1uyrsHYXcJ2HJdDjXuM8RzP5Sb7n04u5y/view?usp=sharing.

## Prerequisites and Installation

1. R version >= 3.6.
2. R packages: Rcpp (>= 1.0.3), RcppArmadillo (>= 0.9.800.1).
3. Install the package SCSC.
```
devtools::install_github("WgitU/SCSC")
```

## Example Code

``` {r, eval=FALSE}
library(SCSC)

#import example data
data(example_data)

#gene number
nrow(count_data_matr)

#cell number
ncol(count_data_matr)

#subject number
length(vec_ncell_subj)

#run SCSC
t1 <- Sys.time()
Result <- SCSC(count_data_matr, vec_ncell_subj, celltype_upb = 10, subgroup_upb = 10,
      seed = 1, num_threads = 10, num_iterations = 1000, print_label = TRUE)
t2 <- Sys.time()

#time cost
print(t2 - t1)

#Compare the estimates with true subject subgroup labels
table(Result$subject_subgroup_label, subject_subgroup_label_truth)

#Compare the estimates with true cell type labels
cell_table <- table(Result$cell_type_label, cell_type_label_truth)
cell_table

#The following shows the summary of the absolute errors of estimated subject subgroup effects
#across genes within each subject subgroup
summary(abs(Result$subject_subgroup_effects - subject_subgroup_effects_truth))

#The following shows the summary of the absolute errors of estimated cell type effects
#across genes within each cell type
type_name <- rownames(which(cell_table > 0,TRUE))
cell_unique <- unique(Result$cell_type_label)
summary(abs(Result$cell_type_effects[,c(which(type_name[1]==cell_unique)
            ,which(type_name[2]==cell_unique), which(type_name[3]==cell_unique))]
             - cell_type_effects_truth))
```
or you can simply run
``` {r, eval=FALSE}
library(SCSC)
example(SCSC)
```
## Remarks
* This package applies openmp to parallel computing. 
* This package can be downloaded and run in Windows and Linux. However, as Mac OS does not support openmp, the package temporarily does not support Mac OS.
* If you have any questions regarding this package, please contact Qiuyu Wu at w.qy@ruc.edu.cn.

