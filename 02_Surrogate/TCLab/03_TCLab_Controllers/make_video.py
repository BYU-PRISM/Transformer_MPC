import imageio  # required to make animation

# generate mp4 from png figures in batches of 350


window = 5
ns = 12*60
images = []
iset = 0
for i in range(244,ns):
    filename='./figures/plot_'+str(i+10000)+'.png'
    images.append(imageio.imread(filename))
    if ((i+1)%1000)==0:
        imageio.mimsave('results_'+str(iset)+'.mp4', images)
        iset += 1
        images = []
if images!=[]:
    imageio.mimsave('results_'+str(iset)+'.mp4', images)