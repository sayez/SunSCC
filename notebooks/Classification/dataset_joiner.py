import os
import numpy as np
import time

root_dir = "../datasets/Classification_dataset/2002-2019"

all_samples ={'train':{},'val':{},'test':{}}
for p in all_samples.keys():
    print('loading', p)
    st = time.time()    
    filename = os.path.join(root_dir,'test',f'all_samples_{p}.npy' )
    tmp = np.load(filename, allow_pickle=True).item()
    print('Elapsed time', time.time()-st)
    
    all_samples[p] = tmp

print("Dumping")
st = time.time()    
tot_npy_file = os.path.join(root_dir, 'test', f'all_samples.npy')
np.save(tot_npy_file, all_samples)
print('Elapsed time', time.time()-st)
