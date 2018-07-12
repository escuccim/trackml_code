import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

## my library ---
from common import *
from draw import *
from utility.file import *
from utility.draw import *
from net.rate   import *
from net.metric import *


## other library ---
from dataset.trackml.score  import score_event
from dataset.others import *
from sklearn.cluster.dbscan_ import dbscan
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

REMAP_LAYER_ID ={
    (12,  2): 0,   (16,  2): 0,    (14,  2): 0,   (18,  2): 0,
    (12,  4): 1,   (16,  4): 1,    (14,  4): 1,   (18,  4): 1,
    (12,  6): 2,   (16,  6): 2,    (14,  6): 2,   (18,  6): 2,
    (12,  8): 3,   (16,  8): 3,    (14,  8): 3,   (18,  8): 3,
    (12, 10): 4,   (16, 10): 4,    (14, 10): 4,   (18, 10): 4,
    (12, 12): 5,   (16, 12): 5,    (14, 12): 5,   (18, 12): 5,
    (12, 14): 6,   (16, 14): 6,    (14, 14): 6,   (18, 14): 6,

    ( 7,  2): 0,   ( 9,  2): 0,
    ( 7,  4): 1,   ( 9,  4): 1,
    ( 7,  6): 2,   ( 9,  6): 2,
    ( 7,  8): 3,   ( 9,  8): 3,
    ( 7, 10): 4,   ( 9, 10): 4,
    ( 7, 12): 5,   ( 9, 12): 5,
    ( 7, 14): 6,   ( 9, 14): 6,
    ( 7, 16): 7,   ( 9, 16): 7,

    (13,  2): 0,
    (13,  4): 1,
    (13,  6): 2,
    (13,  8): 3,
    (17,  2): 4,
    (17,  4): 5,

    ( 8,  2): 0,
    ( 8,  4): 1,
    ( 8,  6): 2,
    ( 8,  8): 3
}
LAYER_NUM = 7




##########################################################33
def make_data(
    a,zr,z, my_layer_id, p,
    # a_limit=(1.0,3.0), zr_limit=(4.0,7.0),
    a_limit=(1.0,2.0), zr_limit=(4.0,5.0),
    depth = 6
):
    a0,  a1 = a_limit
    zr0,zr1 = zr_limit

    idx = np.where( (a>=a0) & (a<a1) & (zr>=zr0) & (zr<zr1) )[0]
    aa,zzr,zz = a[idx], zr[idx],  z[idx]/1000
    ll = my_layer_id[idx]
    pp = p[idx]

    data3 = np.column_stack((aa,zzr,zz))
    L = len(data3)

    pairs=[]
    for d in range(depth-1):
        i0 = np.where(ll==d  )[0]
        i1 = np.where(ll==d+1)[0]

        L0 = len(i0)
        L1 = len(i1)
        if L0==0: continue
        if L1==0: continue

        q0 = data3[i0]
        q1 = data3[i1]
        qq0 = np.repeat(q0.reshape(L0,1,3),L1, axis=1).reshape(-1,3)
        qq1 = np.repeat(q1.reshape(1,L1,3),L0, axis=0).reshape(-1,3)
        ii0 = np.repeat(i0.reshape(L0,1),L1, axis=1).reshape(-1,1)
        ii1 = np.repeat(i1.reshape(1,L1),L0, axis=0).reshape(-1,1)


        unit = qq1-qq0
        unit = unit/np.sqrt((unit**2).sum(1,keepdims=True))
        ii   = np.zeros((L0*L1,1),np.int32)

        pair = np.concatenate((ii0,ii1,ii, qq0,qq1,unit),1)
        pairs.append(pair)

    P = len(pairs)
    M = 0
    for p in pairs:
        dM = len(p)
        p[:,2] = np.arange(M, M+dM)
        M += dM


    distance = np.full((M,M), 100, np.float32) #INF
    for d in range(P-1):
        for a in pairs[d]:
            ai0, ai1, ai = a[:3].astype(np.int32)
            ap,aq,aunit  = np.split(a[3:],3)
            if ((np.fabs(aunit[0])>0.25) | (np.fabs(aunit[1])>0.25)): continue

            b   = pairs[d+1]
            i = (np.where((b[:,0]==ai1)))[0]

            bi  = (b[:,2][i]).astype(np.int32)
            dis = np.sqrt(((b[:,-3:][i]-aunit)**2).sum(1))
            distance[ai,bi] = dis

    print('dbscan')
    _,l = dbscan(distance, eps=0.080, min_samples=1, metric='precomputed')
    cluster_id = np.unique(l+1)
    #cluster_id = cluster_id[cluster_id!=0]
    num_cluster_id = len(cluster_id)

    ## draw clustering results -----------------------------------
    print('draw clustering results')
    pairs_flat = np.vstack(pairs)

    AX3d1.clear()
    AX3d1.scatter(aa, zzr,  zz, c=plt.cm.gnuplot( ll/depth ), s=16, edgecolors='none')
    plot3d_particles(AX3d1, aa, zzr, zz, zz, pp, subsample=1,color=[0,0,0], linewidth=4)

    for id in cluster_id:
        #AX3d1.clear()
        #AX3d1.scatter(aa, zzr,  zz, c=plt.cm.gnuplot( ll/depth ), s=16, edgecolors='none')

        t  = np.where(l==id)
        t0 = pairs_flat[t,0].astype(np.int32).reshape(-1)
        t1 = pairs_flat[t,1].astype(np.int32).reshape(-1)
        t  = np.unique(np.concatenate((t0, t1)))
        #if len(t0)<3: continue

        color = np.random.uniform(0,1,3)
        #AX3d1.plot(aa[t0], zzr[t0],  zz[t0],'.-', color=color, markersize=15) #edgecolors=
        #AX3d1.plot(aa[t1], zzr[t1],  zz[t1],'.-', color=color, markersize=15)
        AX3d1.plot(aa[t], zzr[t],  zz[t],'.-', color=color, markersize=15)
        #plt.pause(0.01)
        #plt.waitforbuttonpress(-1)
    plt.show()


    return 0




def run_study_cnn ():

    # event_ids = ['000001030', '000001029', '000001028', '000001027', '000001026', '000001025'] #
    # event_id = '000001001'
    #event_ids = ['%09d'%i for i in range(1000,1100)]
    #for event_id in event_ids:


    if 1:
        event_id  = '000001029'

        save_dir  = '/media/ssd/data/kaggle/cern'
        data_dir  = '/root/share/project/kaggle/cern/data/__download__/train_100_events'
        particles = pd.read_csv(data_dir + '/event%s-particles.csv'%event_id)
        hits      = pd.read_csv(data_dir + '/event%s-hits.csv' %event_id)
        truth     = pd.read_csv(data_dir + '/event%s-truth.csv'%event_id)

        truth = truth.merge(hits,       on=['hit_id'],      how='left')
        truth = truth.merge(particles,  on=['particle_id'], how='left')


        #----------------------------
        df = truth  #.copy()
        #df = df.assign(my_layer_id  = df.layer_id.replace(REMAP_LAYER_ID, inplace=False))

        df = df.assign(r   = np.sqrt( df.x**2 + df.y**2))
        df = df.assign(d   = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
        df = df.assign(a   = np.arctan2(df.y, df.x))
        df = df.assign(cosa= np.cos(df.a))
        df = df.assign(sina= np.sin(df.a))
        df = df.assign(zr  = df.z/df.r)
        df = df.assign(phi = np.arctan2(df.z, df.r))
        df = df.assign(momentum = np.sqrt( df.px**2 + df.py**2 + df.pz**2 ))
        df = df.assign(vel      = np.sqrt( df.vx**2 + df.vy**2 ))
        df.loc[df.particle_id==0,'momentum']=0

        #top volume_id = (14,18,);  layer_id = (2, 4, 6, 8, 10, 12,)
        my_layers = [0,1,2,3,4,5,]
        df = df.loc[(df.z>1100)& (df.z<INF)]
        df = df.loc[(df.r> 200)& (df.r<INF)]

        #-------------------------------------------------------
        df = df.assign(my_layer_id  = df[['volume_id','layer_id']].apply(lambda x: REMAP_LAYER_ID[tuple(x)], axis=1))
        my_layer_id  = df['my_layer_id'].values.astype(np.int32)
        p            = df['particle_id'].values.astype(np.int64)
        x,y,z,r,a,zr = df[['x', 'y', 'z', 'r', 'a', 'zr']].values.astype(np.float32).T

        particle_ids = np.unique(p)
        particle_ids = particle_ids[particle_ids!=0]
        num_particle_ids = len(particle_ids)


        #-------------------------------------------------------
        tic = time.time()
        make_data( a,zr,z, my_layer_id, p,)
        toc = time.time()

        print('toc-tic Elapsed: %s'%(toc-tic))


        # np.save('/media/ssd/data/kaggle/cern/input.npy',input)
        # np.save('/media/ssd/data/kaggle/cern/truth.npy',truth)
        # np.save('/media/ssd/data/kaggle/cern/index.npy',index)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_study_cnn()


#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#  convert *.png animated.gif
#
