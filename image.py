import cv2, os

name = "aadish"
path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save an image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Capture - Press S to Save", frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord('s'):
        count += 1
        cv2.imwrite(f"{path}/{count}.jpg", frame)
        print(f"Saved {count}.jpg")
    elif k & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
