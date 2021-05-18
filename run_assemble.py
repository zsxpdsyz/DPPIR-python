import os
start = 2
end = 51
interval = 2
for sigma in range(start, end, interval):
    print(sigma)
    os.system('python train.py --sigma '+str(sigma))