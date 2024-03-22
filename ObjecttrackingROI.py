import cv2
import numpy as np

roi_x, roi_y, roi_width, roi_height = 700, 400, 400, 300

num_rows, num_cols = 8, 8
grid_width = roi_width // num_cols
grid_height = roi_height // num_rows
frame_count = 0

result_matrix = np.zeros((num_rows, num_cols), dtype=int)

cap = cv2.VideoCapture('traffic2.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    if not ret:
        break

    frame_count += 1
    if frame1.shape[:2] == frame2.shape[:2]:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.rectangle(frame1, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

        for i in range(1, num_cols):
            cv2.line(frame1, (roi_x + i * grid_width, roi_y), (roi_x + i * grid_width, roi_y + roi_height), (0, 255, 0), 2)
        for j in range(1, num_rows):
            cv2.line(frame1, (roi_x, roi_y + j * grid_height), (roi_x + roi_width, roi_y + j * grid_height), (0, 255, 0), 2)

        result_matrix.fill(0)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 700:
                continue

            if x >= roi_x and y >= roi_y and x + w <= roi_x + roi_width and y + h <= roi_y + roi_height:
                start_grid_row = max(0, (y - roi_y) // grid_height)
                end_grid_row = min(num_rows - 1, ((y + h) - roi_y) // grid_height)
                start_grid_col = max(0, (x - roi_x) // grid_width)
                end_grid_col = min(num_cols - 1, ((x + w) - roi_x) // grid_width)

                color = (0, 255, 0)  
                for grid_row in range(start_grid_row, end_grid_row + 1):
                    for grid_col in range(start_grid_col, end_grid_col + 1):
                        cv2.rectangle(frame1, (roi_x + grid_col * grid_width, roi_y + grid_row * grid_height),
                                      (roi_x + (grid_col + 1) * grid_width, roi_y + (grid_row + 1) * grid_height),
                                      color, -1)

                        result_matrix[grid_row, grid_col] = 1
                        print("Result matrix ")
                        print(result_matrix)

                        cv2.putText(frame1, "Detection in Grid({}, {})".format(grid_row, grid_col),
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        out.write(frame1)

        frame1 = frame2
        ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(frame_count)