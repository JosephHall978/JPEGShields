import watermarkbench as wb
import cv2 as cv
import numpy as np
import os

test = cv.imread('logo.png')
cv.imwrite('./test/image.jpg', test)


#ATTACKING

#- rotate:
out_path = wb.attack.rotate("./test/image.jpg", 15) # rotates ./test/image.jpg by 10 degrees

#- crop:
out_path = wb.attack.crop("./test/image.jpg", 10) # crops ./test/image.jpg by 10 percent

#- scaled:
out_path = wb.attack.scaled("./test/image.jpg", 0.5) # scales ./test/image.jpg by 0.5x

#- flipping:
out_path = wb.attack.flipping("./test/image.jpg", "H") # flips ./test/image.jpg horizontally

#- jpeg:
out_path = wb.attack.jpeg("./test/image.jpg", 50) # compresses ./test/image.jpg by Q = 50

#- jpeg2000:
out_path = wb.attack.jpeg2000("./test/image.jpg", 10) # compresses ./test/image.jpg by Q = 10

#- jpegai: COMPRESSAI
#out_path = wb.attack.jpegai("./test/image.jpg", 1) # compresses ./test/image.jpg by Q = 1

#- jpegxl: PILLOW-JXL
#out_path = wb.attack.jpegxl("./test/image.jpg", 12) # compresses ./test/image.jpg by D = 12

#- gauissian noise:
out_path = wb.attack.gaussian_noise("./test/image.jpg", 0.01) # adds noise to ./test/image.jpg by σ = 0.01

#- speckle noise:
out_path = wb.attack.speckle_noise("./test/image.jpg", 0.3) # adds noise to ./test/image.jpg by σ = 0.3

#- blurring:
out_path = wb.attack.blurring("./test/image.jpg", 5.0) # blurs ./test/image.jpg by k = 5 x 5

#- brightness:
out_path = wb.attack.brightness("./test/image.jpg", 1.3) # brightens ./test/image.jpg by factor = 1.3

#- sharpness:
out_path = wb.attack.sharpness("./test/image.jpg", 1.25) # sharpens ./test/image.jpg by 1.25x

#- median filtering:
out_path = wb.attack.median_filtering("./test/image.jpg", 5) # median filters ./test/image.jpg by k = 5 x 5

#- ai removal: weird error
#out_path = wb.attack.remove_ai("./test/image.jpg") # uses AI to remove an object in ./test/image.jpg

#- ai replacement: NO KEY FOR NOW
#os.environ["OPENAI_API_KEY"] = "sk......."
#out_path = wb.attack.replace_ai("./test/image.jpg") # uses AI to replace an object in ./test/image.jpg

#- ai creation: NO KEY FOR NOW
#os.environ["OPENAI_API_KEY"] = "sk......."
#out_path = wb.attack.create_ai("./test/image.jpg") # uses AI to create an object/objects in ./test/image.jpg


#EMBEDDING METRICS:
import WaterMarker as wm
mark = np.random.randint(0, 2, size=100, dtype=np.uint8)
marker = wm.WaterMarker()
test = cv.imread('logo.png')
cv.imwrite('./test/original.png', test)
marked = marker.generate(test,mark)
cv.imwrite('./test/watermarked.png', marked)

ssim = wb.embedding.SSIM("./test/original.png", "./test/watermarked.png") # calculates SSIM score
psnr = wb.embedding.PSNR("./test/original.png", "./test/watermarked.png") # calculates PSNR score
wpsnr = wb.embedding.WPSNR("./test/original.png", "./test/watermarked.png") # calculates wPSNR score
mse = wb.embedding.MSE("./test/original.png", "./test/watermarked.png") # calculates MSE score
jnd = wb.embedding.JNDPassRate("./test/original.png", "./test/watermarked.png") # calculates JND score

print(ssim)
print(psnr)
print(wpsnr)
print(mse)

#EXTRACTING METRICS:

ber = wb.extracting.BER("010101", "010001") # calculates bit error between ground truth and extracted watermark
print(ber)  # expected 1/6 bit