import os, sys, subprocess


path_base = "/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/KTH/"
# for i in range(25):
# 	os.mkdir(path_base+"boxing/p"+str(i))
# 	os.mkdir(path_base+"handwaving/p"+str(i))
# 	os.mkdir(path_base+"handclapping/p"+str(i))

# 	for k in range(1,5):
# 		os.mkdir(path_base+"boxing/p"+str(i)+"/d"+str(k))
# 		os.mkdir(path_base+"handwaving/p"+str(i)+"/d"+str(k))
# 		os.mkdir(path_base+"handclapping/p"+str(i)+"/d"+str(k))

# for i in range(0,25):
# 	for k in range(1,5):
# 		os.mkdir(path_base+str(i)+"/d"+str(k))


# for i in range(25):
# 	# os.remove(path_base+str(i))
# 	subprocess.call("rm "+path_base+str(i)+"/*", shell=True)

path_base = "/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/KTH/boxing/"
for i in range(1,26):
	for k in range(1,5):
		subprocess.call("ffmpeg -i " + path_base + "person"+str(i).zfill(2)+"_boxing_d"+str(k)+"_uncomp.avi " + path_base + "p"+str(i-1)+"/d"+str(k)+"/%d.png", shell=True)

path_base = "/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/KTH/handwaving/"
for i in range(1,26):
	for k in range(1,5):
		subprocess.call("ffmpeg -i " + path_base + "person"+str(i).zfill(2)+"_handwaving_d"+str(k)+"_uncomp.avi " + path_base + "p"+str(i-1)+"/d"+str(k)+"/%d.png", shell=True)


path_base = "/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/KTH/handclapping/"
for i in range(1,26):
	for k in range(1,5):
		subprocess.call("ffmpeg -i " + path_base + "person"+str(i).zfill(2)+"_handclapping_d"+str(k)+"_uncomp.avi " + path_base + "p"+str(i-1)+"/d"+str(k)+"/%d.png", shell=True)
