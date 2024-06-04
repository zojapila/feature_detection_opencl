import cv2



def resize_image_cv2(image, target_width):
    height, width = image.shape[:2]
    aspect_ratio = target_width / float(width)
    target_height = int(height * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    cv2.imwrite("FeatureDetection_Inputx.bmp", resized_image)
    return resized_image

im= cv2.imread("/home/zosia/aaw/BARCODE_0015.bmp")
print(im)
resize_image_cv2(im, 4096)
