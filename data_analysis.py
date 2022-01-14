import pandas as pd
from rpy2.robjects import r, pandas2ri
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm

from tqdm import tqdm
from sklearn import model_selection

path = r'C:\Users\niels\OneDrive\Skrivebord\Statistisk Analyse af Kunstig Intelligens\projekt\\'

robj = r.load(path + 'armdata.RData')


#%% get data

# eksperiment person repitition time dimension
pos_data = np.array(r['armdata'])

# there's 12*3 missing datapoints
# it's all the first 1, 2, or 4 datapoints i a series in 6 of the trails
# nan_data = np.isnan(pos_data)
# nan_indexes = np.where(nan_data)

# pd.DataFrame(nan_indexes).to_csv('nan_indexes.csv')

# the nan values are replaced by copying the first non nan value in the 
# dataseries

pos_data[4,8,0,0,0:3] = pos_data[4,8,0,1,0:3]
pos_data[6,8,1,0,0:3] = pos_data[6,8,1,1,0:3]
pos_data[9,8,0,0:2,0:3] = pos_data[9,8,0,2,0:3]
pos_data[10,8,0,0:2,0:3] = pos_data[10,8,0,2,0:3]
pos_data[12,8,0,0:4,0:3] = pos_data[12,8,0,4,0:3]
pos_data[13,8,1,0:2,0:3] = pos_data[13,8,1,2,0:3]

# get velecity and acceleration data
# vel_data = np.diff(pos_data, n=1, axis=3)
# acc_data = np.diff(pos_data, n=2, axis=3)

# calculate mean trajectories
# mean_pos = np.mean(pos_data, axis=(1,2))
# mean_vel = np.mean(vel_data, axis=(1,2))
# mean_acc = np.mean(acc_data, axis=(1,2))

# MES of test statistics
# pos_test_stat = np.mean(np.array([(e1 - e2)**2 for e1, e2 in zip(pos_data, mean_pos)])**2, axis=3)
# vel_test_stat = np.mean(np.array([(e1 - e2)**2 for e1, e2 in zip(vel_data, mean_vel)])**2, axis=3)
# acc_test_stat = np.mean(np.array([(e1 - e2)**2 for e1, e2 in zip(acc_data, mean_acc)])**2, axis=3)


#%%

if __name__ == "__main__":
    
    distance_forward = np.diff(pos_data, n=1, axis=3)[:,:,:,:,0][np.diff(pos_data, n=1, axis=3)[:,:,:,:,0] >= 0].sum()
    distance_backward = np.diff(pos_data, n=1, axis=3)[:,:,:,:,0][np.diff(pos_data, n=1, axis=3)[:,:,:,:,0] < 0].sum()
    
    percent_distance_backward = abs(distance_backward) / (distance_forward + abs(distance_backward))


#%%
'''
def yz(x, i):
    xs = pos_data[i[0],i[1],i[2],:,0]
    ys = pos_data[i[0],i[1],i[2],:,1]
    zs = pos_data[i[0],i[1],i[2],:,2]
    
    i0 = np.where(xs <= x)[0][-1]
    i1 = i0 + 1
    
    x_factor = (x - xs[i0]) / (xs[i1] - xs[i0])
    
    y = x_factor * (ys[i1] - ys[i0]) + ys[i0]
    z = x_factor * (zs[i1] - zs[i0]) + zs[i0]

    return y, z


largest_smallest_x = max(np.amin(pos_data, axis=3)[:,:,:,0].flatten())
smallest_largest_x = min(np.amax(pos_data, axis=3)[:,:,:,0].flatten())

number_xs = 100
xs2compare = np.linspace(largest_smallest_x, smallest_largest_x, num=number_xs,
                      endpoint=False)

similarity_data = np.zeros(pos_data.shape[:3] + (number_xs, 2))

for i0, sim1 in enumerate(similarity_data):
    for i1, sim2 in enumerate(sim1):
        for i2, sim3 in enumerate(sim2):
            i = (i0,i1,i2)
            similarity_data[i] = np.array([yz(x, i) for x in xs2compare])

#%%

gen_error_person = []
for p in range(10):
    for e1 in range(15):
        for e2 in range(15):
            gen_error_person += [np.mean([np.linalg.norm(similarity_data[e1,p,i1]-similarity_data[e2,p,i2]) for i1 in range(10) for i2 in range(10)])]


#%%

gen_error_eksperiment = []
for e in range(15):
    for p1 in range(10):
        for p2 in range(10):
            gen_error_eksperiment += [np.mean([np.linalg.norm(similarity_data[e,p1,i1]-similarity_data[e,p2,i2]) for i1 in range(10) for i2 in range(10)])]
'''


#%%
exp2distance = {i: i%5 for i in range(15)}
exp2distance[15] = 5

exp2obstacle = {i: i%3 for i in range(15)}
exp2obstacle[15] = 4

exp_ = []
prs_ = []
rep_ = []
data_ = []
for e in range(16):
    for p in range(10):
        for r_ in range(10):
            exp_ += [e]
            prs_ += [p]
            rep_ += [r]
            data_ += [pos_data[e,p,r_,:,:]]

dis_ = [exp2distance[exp] for exp in exp_]
obs_ = [exp2obstacle[exp] for exp in exp_]


lab_ = list(range(1600))

data_frame = pd.DataFrame(data = {'lab': lab_, 'exp': exp_, 'dis': dis_, 
                                  'obs': obs_,'prs': prs_, 'rep': rep_,
                                  'data': data_})

#%%

curve_i = data_frame[ data_frame['exp'] != 15 ]
curve_mu = np.mean(curve_i['data'])

residuals = np.array([curve_i['data'][i] - curve_mu for i in range(1500)])


#%%

if __name__ == "__main__":
    
    num2axis = {0: 'x', 1: 'y', 2: 'z'}
    
    for i in tqdm(range(100)):
        for axis in range(3):
            fig_name = path + f'pics/{num2axis[axis]}_{str(i).zfill(3)}.png'
            sm.qqplot(residuals[:,i,axis], fit=True, line='45').savefig(fig_name)


#%%

if __name__ == "__main__":
    
    num2axis = {0: 'x', 1: 'y', 2: 'z'}
    
    for i in tqdm(range(100)):
        for num, cat in [(10, 'prs'), (3, 'obs'), (5, 'dis')]:
            cat_data = data_frame[cat][data_frame['exp'] != 15 ]
            for axis in range(3):
                for j in range(num):
                    fig_name = path + f'pics/{cat}/{str(j).zfill(2)}_{num2axis[axis]}_{str(i).zfill(3)}.png'
                    data = residuals[ cat_data == j ][:,i,axis]
                    sm.qqplot(data, fit=True, line='45').savefig(fig_name)


#%%

if __name__ == "__main__":
    
    for i in tqdm(range(300)):
        fig_name = path + f'test_pics/test_pic_{str(i).zfill(3)}.png'
        
        data = np.random.normal(0, 1, 1500)
        
        sm.qqplot(data, fit=True, line='45').savefig(fig_name)


#%%

curve_dis = np.array([np.mean(curve_i[ curve_i['dis'] == i ]['data']) for i in range(5)]) - curve_mu
curve_obs = np.array([np.mean(curve_i[ curve_i['obs'] == i ]['data']) for i in range(3)]) - curve_mu
curve_prs = np.array([np.mean(curve_i[ curve_i['prs'] == i ]['data']) for i in range(10)]) - curve_mu


SST = np.array([(curve_i['data'][i] - curve_mu)**2 for i in range(1500)]).sum(axis=0)

SS_dis = 1500/5 * (curve_dis**2).sum(axis=0)
SS_obs = 1500/3 * (curve_obs**2).sum(axis=0)
SS_prs = 1500/10 * (curve_prs**2).sum(axis=0)

SSE = SST - (SS_dis + SS_obs + SS_prs)

F_dis = (SS_dis/(5-1)) / (SSE/(1500 - (5-1) - (3-1) - (10-1) - 1))
F_obs = (SS_obs/(3-1)) / (SSE/(1500 - (5-1) - (3-1) - (10-1) - 1))
F_prs = (SS_prs/(10-1)) / (SSE/(1500 - (5-1) - (3-1) - (10-1) - 1))

p_dis = 1 - stats.f.cdf(F_dis, 5-1, 1500 - (5-1) - (3-1) - (10-1) - 1)
p_obs = 1 - stats.f.cdf(F_obs, 3-1, 1500 - (5-1) - (3-1) - (10-1) - 1)
p_prs = 1 - stats.f.cdf(F_prs, 10-1, 1500 - (5-1) - (3-1) - (10-1) - 1)


#%%

levene_dis_list = np.array([residuals[ curve_i['dis'] == i ] for i in range(5)])
levene_obs_list = np.array([residuals[ curve_i['obs'] == i ] for i in range(3)])
levene_prs_list = np.array([residuals[ curve_i['prs'] == i ] for i in range(10)])

levene_dis_p = np.array([[stats.levene(*levene_dis_list[:,:,t,d])[1] for d in range(3)] for t in range(100)])
levene_obs_p = np.array([[stats.levene(*levene_obs_list[:,:,t,d])[1] for d in range(3)] for t in range(100)])
levene_prs_p = np.array([[stats.levene(*levene_prs_list[:,:,t,d])[1] for d in range(3)] for t in range(100)])


#%%












