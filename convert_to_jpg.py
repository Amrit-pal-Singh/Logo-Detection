from PIL import Image 
import glob
for file in glob.glob("*.png"):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    rgb_im.save(file.replace("png", "jpg"), quality=95)

for file in glob.glob("*.jpeg"):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    rgb_im.save(file.replace("jpeg", "jpg"), quality=95)

for file in glob.glob("*.webp"):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    rgb_im.save(file.replace("jpeg", "jpg"), quality=95)

for file in glob.glob("*.JPG"):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    rgb_im.save(file.replace("JPG", "jpg"), quality=95)
