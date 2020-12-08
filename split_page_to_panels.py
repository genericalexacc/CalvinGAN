from scipy import misc
import imageio
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir('./good/') if (isfile(join('./good/', f)) and (not f == '.DS_Store'))]

count = 1
for x in onlyfiles:
	f = './good/'+x
	print(f)
	img = imageio.imread(f)
	try:
		height, width, _ = img.shape
	except Exception as e:
		continue

	width_cutoff = width // 4

	s1 = img[:, :width_cutoff]
	s2 = img[:, width_cutoff:width_cutoff*2]
	s3 = img[:, width_cutoff*2:width_cutoff*3]
	s4 = img[:, width_cutoff*3:]

	imageio.imsave("./cut/" + str(count) + "1.jpg", s1)
	imageio.imsave("./cut/" + str(count) + "2.jpg", s2)
	imageio.imsave("./cut/" + str(count) + "3.jpg", s3)
	imageio.imsave("./cut/" + str(count) + "4.jpg", s4)

	count += 1