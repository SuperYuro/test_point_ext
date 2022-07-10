import cv2
from IPython.display import Image, display


def imshow(img):
    ret, encoded = cv2.imencode(".png", img)
    display(Image(encoded))


def detect_feature(img):
    detector = cv2.AKAZE_create()
    kp = detector.detect(img)
    dst = cv2.drawKeypoints(img, kp, None)
    cv2.imshow("Detected Points", dst)
    cv2.waitKey(0)


def harris_corner(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_harris = cv2.cornerHarris(img_gray, 2, 1, 0.04)

    img_cp = img.copy()
    img_harris = cv2.dilate(img_harris, None)
    img_cp[img_harris > 0.01 * img_harris.max()] = [0, 255, 0]

    cv2.imshow("Harris Corner", img_cp)
    cv2.waitKey(0)


def main(img):
    harris_corner(img)


if __name__ == "__main__":
    img = cv2.imread("daruma_padd.png")
    main(img)
