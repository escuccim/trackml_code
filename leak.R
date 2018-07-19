require(data.table)
require(bit64)


path='./data/train_100_events/' 
nev<-1001
dft=fread(paste0(path,'event00000',nev,'-truth.csv'))
dfc=fread(paste0(path,'event00000',nev,'-cells.csv'))


dfc[,valS:=ifelse(value==1,sum(value),0),by=hit_id] 
#dfc <- unique(dfc[,.(hit_id,valS)])

setkey(dfc,hit_id)
setkey(dft,hit_id)
dft <- merge(dft,dfc,all.x = TRUE)
dft <- dft[,.(hit_id,particle_id,valS)]
dft[,part0:=ifelse(particle_id==0,1,0)]
dft[,prob0:=mean(part0),by=valS]

cat("Probalility particle_id==0:",mean(dft$part0))

dfX <- unique(dft[,.(valS,prob0)])
setkey(dfX,prob0)

print(head(dfX,100))