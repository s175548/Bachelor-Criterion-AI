import json
from data_import.draw_contours import draw_contours2

seg = json.loads(read_file(segmentation_path).decode("utf-8"))
segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})
pass
for i in range(3):
    print(i**2)
