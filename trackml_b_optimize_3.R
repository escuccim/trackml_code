########################################################
# Bayesian Optimization
#
# updated by Eric Scuccimarra on 2018.07.08 to add
# extra params to optimize
#
# author: Grzegorz Sionkowski
# date:   2018.06.14
#
# Credits to:
# 1. Mikhail Hushchyn and 5 collaborators
# benchmark in Python
# https://www.kaggle.com/mikhailhushchyn/dbscan-benchmark
# 2. Vicens Gaitan
# translating benchmark into R
# https://www.kaggle.com/vicensgaitan/r-scoring-function
# 3. Heng CherKeng
# feature engineering 
# https://www.kaggle.com/sionek/mod-dbscan-x-100-parallel#337932
# ------------------------------------------------------
########################################################

require(data.table)
require(bit64)
require(dbscan)
require(doParallel)
require(rBayesianOptimization)
path='./data/train_100_events/'

#########################
score<-function(sub_dt,dft){
    df=merge(sub_dt,dft[,.(hit_id,particle_id,weight)],by="hit_id")
    df[,Np:=.N,by=particle_id]# Np = Hits per Particle
    df[,Nt:=.N,by=track_id]   # Nt = Hits per Track
    df[,Ntp:=.N,by=list(track_id,particle_id)]# Hits per Particle per Track
    df[,r1:=Ntp/Nt]
    df[,r2:=Ntp/Np]
    sum(df[r1>.5 & r2>.5,weight])
}
######################## 
### working function ###
########################
trackML <- function(dfh,w1=0.95,w2,w3,w4,Niter=221,epsilon=350,step, stepeps, max_size=20, size_incr=0, step_z=0){
    stepeps = stepeps / Niter
    epsilon = epsilon / 100000
    if (size_incr != 0) {
        size_incr = (21 - max_size) / (Niter * size_incr)
    }
    
    dfh[,s1:=hit_id]
    dfh[,N1:=1L] 
    dfh[,r:=sqrt(x*x+y*y+z*z)]
    dfh[,rt:=sqrt(x*x+y*y)]
    dfh[,a0:=atan2(y,x)]
    dfh[,z1:=z/rt]
    dfh[,z2:=z/r]
    dfh[,z3:=1/z]
    dfh[,x2:=x/r]
    dfh[,y2:=x/r]
    dfh[,stepped_z:=z]
    
    mm     <-  1
    dz0 = -0.00070
    stepdz = 0.00001
    
    for (ii in 0:Niter) {
        max_cluster_size = max(c(max_size + (ii * size_incr), 21))
        mm <- mm*(-1)
        # dz = mm + dz0 * stepdz
        dfh[,a1:=a0+mm*(rt+(step*ii)*rt^2)/1000*(ii/2)/180*pi]
        dfh[,sina1:=sin(a1)]
        dfh[,cosa1:=cos(a1)]
        dfs=scale(dfh[,.(sina1, cosa1, z1, z2, x2, y2)])
        cx <- c(w1,w1,w2,w3,w4,w4)
        for (jj in 1:ncol(dfs)) dfs[,jj] <- dfs[,jj]*cx[jj]
        res=dbscan(dfs,eps=epsilon+(ii*stepeps),minPts = 1)
        dfh[,s2:=res$cluster]
        dfh[,N2:=.N, by=s2]
        maxs1 <- max(dfh$s1)
        dfh[,s1:=ifelse(N2>N1 & N2<max_cluster_size,s2+maxs1,s1)]
        dfh[,s1:=as.integer(as.factor(s1))]
        dfh[,N1:=.N, by=s1]    
        
        # step z and recalculate columns
        if(step_z != 0){
            dfh[,stepped_z:=z + (mm * step_z * ii)]
            dfh[,r:=sqrt(x*x+y*y+stepped_z*stepped_z)]
            dfh[,z2:=stepped_z/r]
            dfh[,x2:=x/r]
            dfh[,y2:=x/r]    
        }
    }
    return(dfh$s1)
}
#######################################
# function for Bayessian Optimization #
#   (needs lists: Score and Pred)     #
#######################################
Fun4BO <- function(w1=0.95,w2,w3,w4,Niter=221,epsilon=350,step, stepeps, max_size=20, size_incr=0, step_z=0) { 
    dfh$s1 <- trackML(dfh,w1,w2,w3,w4,Niter,epsilon,step, stepeps, max_size, size_incr, step_z)
    sub_dt=data.table(event_id=nev,hit_id=dfh$hit_id,track_id=dfh$s1)
    sc <- score(sub_dt,dft)
    list(Score=sc,Pred=0)
}
################################### 
###    Bayesian Optimization    ###
###################################
print("Bayesian Optimization")
nev=1003
dfh=fread(paste0(path,'event00000',nev,'-hits.csv'), showProgress=F)
dft=fread(paste0(path,'event00000',nev,'-truth.csv'),stringsAsFactors = T, showProgress=F)
OPT <- BayesianOptimization(Fun4BO,
                            # bounds = list(w1 = c(0.9, 1.2), w2 = c(0.3, 0.5), w3 = c(0.2, 0.4), w4 = c(0.0, 0.1),Niter = c(150L, 190L), epsilon=c(350L, 351L), step=c(0.000000, 0.000005), stepeps=c(0,0.00001)),
                            # bounds = list(w1 = c(0.9, 1.2), w2 = c(0.3, 0.5), w3 = c(0.15, 0.25), w4 = c(0.0, 0.01),Niter = c(150L, 151L), epsilon=350, step=c(0.000000, 0.00003), stepeps=c(0,0.000001), max_size=20, size_incr=0),
                            # bounds = list(w1 = c(0.9, 1.2), w2 = c(0.2, 0.5), w3 = c(0.1, 0.4), w4 = c(0.0, 0.01),Niter = c(150L, 151L), step=c(0.000000, 0.00003), stepeps=c(0,0.000001)),
                            #bounds = list(w1 = c(0.9, 1.0), w2 = c(0.3, 0.5), w3 = c(0.1, 0.4), w4 = c(0.05, 0.02), Niter = c(150L, 200L), epsilon=c(340L, 350L), step=c(0.000000, 0.000001), stepeps=c(0,0.001), step_z=c(0, 0.00001)),
                            bounds = list(w1 = c(0.9, 1.1),w2 = c(0.3, 0.5), w3 = c(0.03, 0.3), w4 = c(0.0, 0.015), epsilon=c(340L, 350L), step=c(0.000000, 0.000001), stepeps=c(0.0,0.00001), max_size=c(16L,21L), step_z=c(0.000005, 0.00003), size_incr=c(0.9, 1.1)),
                            init_points = 3, n_iter = 40,
                            acq = "ucb", kappa = 2.576, eps = 0.0,
                            verbose = TRUE)

########################
###    submission    ###
########################
namef <- system("ls ./data/test/*hits.csv", intern=TRUE)
#path <- './data/test/'
print("Preparing submission")
w1    <- OPT$Best_Par[[1]]
w2    <- OPT$Best_Par[[2]]
w3    <- OPT$Best_Par[[3]]
w4    <- OPT$Best_Par[[4]]
Niter <- OPT$Best_Par[[5]]
epsilon <- OPT$Best_Par[[6]]
step <- OPT$Best_Par[[7]]
stepeps <- OPT$Best_Par[[8]]
max_size <- OPT$Best_Par[[9]]
step_z <- OPT$Best_Par[[10]]
size_inc <- OPT$Best_Par[[11]]

registerDoParallel(cores=4)
print("Parallel")

sub_dt <- foreach(nev = 0:124, .combine = rbind, .export=c("fread", "dbscan", "data.table")) %dopar%  {
    dfh <- fread(namef[nev+1], showProgress=F)
    dfh$s1 <- trackML(dfh,w1,w2,w3,w4,Niter,epsilon,step, stepeps, max_size, size_inc, step_z)
    subEvent <- data.table(event_id=nev,hit_id=dfh$hit_id,track_id=dfh$s1)
    return(subEvent)    
}

fwrite(sub_dt, "sub-Bayes-Opt-DBSCAN_14.csv", showProgress=F)
print('Finished')

