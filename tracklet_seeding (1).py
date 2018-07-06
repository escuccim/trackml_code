import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

## my library ---
from common import *
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



#https://www.kaggle.com/cpmpml/a-faster-python-scoring-function
def cpmp_fast_score(truth, submission):

    truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    df    = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()
    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])

    df1   = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()
    truth = truth.merge(df1, how='left', on='particle_id')
    df1   = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()
    truth = truth.merge(df1, how='left', on='track_id')
    truth.count_both *= 2


    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()
    results = truth


    return score, results

##--------------------------------------------------------


def study_dbscan_for_tracklet_seeding():

    ## load an event ---
    event_id = '000001029'

    data_dir  = '/root/share/project/kaggle/cern/data/__download__/train_100_events'
    particles = pd.read_csv(data_dir + '/event%s-particles.csv'%event_id)
    hits      = pd.read_csv(data_dir + '/event%s-hits.csv' %event_id)
    truth     = pd.read_csv(data_dir + '/event%s-truth.csv'%event_id)
    #cells = pd.read_csv(data_dir + '/event%s-cells.csv'%event_id)

    truth = truth.merge(hits,       on=['hit_id'],      how='left')
    truth = truth.merge(particles,  on=['particle_id'], how='left')

    #--------------------------------------------------------
    df = truth.copy()
    df = df.assign(r   = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(d   = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(a   = np.arctan2(df.y, df.x))
    df = df.assign(cosa= np.cos(df.a))
    df = df.assign(sina= np.sin(df.a))
    df = df.assign(phi = np.arctan2(df.z, df.r))
    df = df.assign(momentum = np.sqrt( df.px**2 + df.py**2 + df.pz**2 ))
    df.loc[df.particle_id==0,'momentum']=0

    df = df.loc[df.z>500] # consider dataset subset
    df = df.loc[df.r<50 ]
    N = len(df)

    #-------------------------------------------------------
    momentum = df[['momentum']].values.astype(np.float32)
    p = df[['particle_id']].values.astype(np.int64)
    x,y,z,r,a,cosa,sina,phi = df[['x', 'y', 'z', 'r', 'a', 'cosa', 'sina', 'phi']].values.astype(np.float32).T

    particle_ids = np.unique(p)
    particle_ids = particle_ids[particle_ids!=0]
    num_particle_ids = len(particle_ids)

    # do dbscan here =======================================
    data   = np.column_stack([a, z/r*0.1])

    _,l = dbscan(data, eps=0.01, min_samples=1,)

    submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
        data=np.column_stack(([int(event_id),]*len(df), df.hit_id.values, l))
    ).astype(int)
    score1 = score_event(df, submission)
    score2, results = cpmp_fast_score(df, submission)

    #print results
    max_score = df.weight.sum()
    print('max_score = df.weight.sum() = %0.5f'%max_score)
    print('score1= %0.5f  (%0.5f)'%(score1*max_score,score1))
    print('score2= %0.5f  (%0.5f)'%(score2,score2/max_score))


    ## analyse the results here =============================
    d0,d1 = data.T
    track_ids = np.unique(l)
    track_ids = track_ids[track_ids!=0]
    num_track_ids = len(track_ids)


    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, )
    fig.patch.set_facecolor('white')

    fig1 = plt.figure(figsize=(8,8))
    ax1  = fig1.add_subplot(111, projection='3d')
    fig1.patch.set_facecolor('white')

    def show_ax():
        ax1.set_xlabel('a', fontsize=16)
        ax1.set_ylabel('r', fontsize=16)
        ax1.set_zlabel('z', fontsize=16)
        ax.set_xlabel('a', fontsize=16)
        ax.set_ylabel('z/r', fontsize=16)
        # ax.grid()
        # ax.set_aspect('equal', 'box')

        plt.show()

    ## 0. show data:
    if 0:
        ax.clear()
        ax1.clear()
        ax.plot (d0,d1, '.',  color=[0.75,0.75,0.75], markersize=3,  linewidth=0)
        ax1.plot(a,r, z,'.',  color=[0.75,0.75,0.75], markersize=3,  linewidth=0)
        show_ax()



    ## 1. show GT:
    if 1:

        ax.clear()
        ax1.clear()
        ax.plot (d0,d1, '.',  color=[0.75,0.75,0.75], markersize=3,  linewidth=0)
        ax1.plot(a,r, z,'.',  color=[0.75,0.75,0.75], markersize=3,  linewidth=0)

        ax1.set_xlabel('a', fontsize=16)
        ax1.set_ylabel('r', fontsize=16)
        ax1.set_zlabel('z', fontsize=16)
        ax.set_xlabel('a', fontsize=16)
        ax.set_ylabel('z/r', fontsize=16)

        for n in range(0,num_particle_ids,1):
            particle_id = particle_ids[n]
            t = np.where(p==particle_id)[0]
            #if momentum[t[0]]<min_momentum: continue
            t = t[np.argsort(np.fabs(z[t]))]

            if np.fabs(a[t[0]]-a[t[-1]])>1: continue
            d = ((x[t[0]]-x[t[-1]])**2 + (y[t[0]]-y[t[-1]])**2 + (z[t[0]]-z[t[-1]])**2)**0.5
            if d<10: continue

            print(n, particle_id)
            color = np.random.uniform(0,1,(3))

            #ax.clear()
            #ax1.clear()

            ax.plot(data[t,0],data[t,1], '.',  color=color, markersize=5,  linewidth=0)
            ax1.plot(a[t],r[t], z[t],'.-',  color=color, markersize=5,  linewidth=1)
            #ax1.plot(a[h],r[h], z[h], 'o',  color=[0,0,0], markersize=8,  linewidth=1, mfc='none')

            #ax1.view_init(0, (ax_n*3)%360)
            #ax_n += 1

            #fig1.savefig('/root/share/project/kaggle/cern/results/yy/%05d.png'%ax_n)
            #plt.pause(0.01)
            #plt.waitforbuttonpress(-1)

        #show_ax()



    ## 2. show dbscan prediction:
    if 1:
        fig_ = plt.figure(figsize=(8,8))
        ax_  = fig_.add_subplot(111, )
        fig_.patch.set_facecolor('white')

        fig1_ = plt.figure(figsize=(8,8))
        ax1_  = fig1_.add_subplot(111, projection='3d')
        fig1_.patch.set_facecolor('white')


        ax_.clear()
        ax1_.clear()
        ax_.plot (d0,d1, '.',  color=[0.75,0.75,0.75], markersize=3,  linewidth=0)
        ax1_.plot(a,r, z,'.',  color=[0.75,0.75,0.75], markersize=3,  linewidth=0)

        ax1_.set_xlabel('a', fontsize=16)
        ax1_.set_ylabel('r', fontsize=16)
        ax1_.set_zlabel('z', fontsize=16)
        ax_.set_xlabel('a', fontsize=16)
        ax_.set_ylabel('z/r', fontsize=16)

        for n in range(0,num_track_ids,1):
            track_id = track_ids[n]
            t = np.where(l==track_id)[0]
            #if momentum[t[0]]<min_momentum: continue
            t = t[np.argsort(np.fabs(z[t]))]

            if np.fabs(a[t[0]]-a[t[-1]])>1: continue
            d = ((x[t[0]]-x[t[-1]])**2 + (y[t[0]]-y[t[-1]])**2 + (z[t[0]]-z[t[-1]])**2)**0.5
            if d<10: continue

            print(n, track_id)
            color = np.random.uniform(0,1,(3))

            #ax.clear()
            #ax1.clear()

            ax_.plot(data[t,0],data[t,1], '.',  color=color, markersize=5,  linewidth=0)
            ax1_.plot(a[t],r[t], z[t],'.-',  color=color, markersize=5,  linewidth=1)
            #ax1.plot(a[h],r[h], z[h], 'o',  color=[0,0,0], markersize=8,  linewidth=1, mfc='none')

            #ax1.view_init(0, (ax_n*3)%360)
            #ax_n += 1

            #fig1.savefig('/root/share/project/kaggle/cern/results/yy/%05d.png'%ax_n)
            #plt.pause(0.01)
            #plt.waitforbuttonpress(-1)

        #show_ax()
        #plt.show()

    ################################################################################################

    # analysis ...
    ## <to be updated> ...

    #shows detected tracks
    #...

    #shows false positive tracks
    #...

    #shows missed tracks
    #...


    zz=0



    exit(0)
    ## kdtree = KDTree(data)
    ## shows nearest neighours  




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    study_dbscan_for_tracklet_seeding()



    print('\nsucess!')



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#  convert *.png animated.gif
#
