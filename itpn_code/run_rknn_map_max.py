import cv2
import numpy as np
from rknn.api import RKNN
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COCODataset:
    def __init__(self, ann_file, img_dir):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_ids = list(sorted(self.coco.imgs.keys()))

        # Load category information
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = tuple([cat['name'] for cat in self.cats])
        self.num_classes = len(self.classes)

        # Create category ID mappings
        self.cat_ids = self.coco.getCatIds()
        self._class_to_coco_ind = dict(zip(self.classes, self.cat_ids))
        self._coco_ind_to_class_ind = dict(
            zip(self.cat_ids, range(self.num_classes)))
        self._class_ind_to_coco_ind = {v: k for k, v in self._coco_ind_to_class_ind.items()}

        logger.info(f"Loaded {self.num_classes} classes")
        logger.info(f"Class to COCO ID mapping: {self._class_to_coco_ind}")

    def __len__(self):
        return len(self.img_ids)

    def get_image_info(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        return image_path, img_info, img_id

    def get_annotations(self, img_id):
        """Get ground truth annotations for an image"""
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def get_coco_category_id(self, class_idx):
        """Convert model class index to COCO category ID"""
        return self._class_ind_to_coco_ind.get(class_idx, -1)


class ObjectDetector:
    def __init__(self, model_path, dataset, target_size=(480, 480), conf_thresh=0.3, nms_thresh=0.45):
        self.target_size = target_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.dataset = dataset

        # Initialize RKNN
        self.rknn = RKNN()

        # Load RKNN model
        logger.info('Loading RKNN model...')
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            logger.error('Load RKNN model failed')
            exit(ret)

        # Initialize runtime environment
        logger.info('Init runtime environment...')
        ret = self.rknn.init_runtime(target='rk3588')
        if ret != 0:
            logger.error('Init runtime environment failed')
            exit(ret)

        logger.info('RKNN model and runtime initialized successfully')

        # Initialize colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    def __del__(self):
        if hasattr(self, 'rknn'):
            self.rknn.release()

    def set_thresholds(self, conf_thresh, nms_thresh):
        """Update confidence and NMS thresholds"""
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def nms(self, boxes, scores, iou_threshold):
        """Non-maximum suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        result = []
        index = scores.argsort()[::-1]

        while index.size > 0:
            i = index[0]
            result.append(i)

            if index.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[index[1:]])
            yy1 = np.maximum(y1[i], y1[index[1:]])
            xx2 = np.minimum(x2[i], x2[index[1:]])
            yy2 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[index[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            index = index[inds + 1]

        return result

    def preprocess(self, cv_image, use_norm=True):
        """Preprocess image for model input"""
        h, w, c = cv_image.shape
        top, bottom, left, right = 0, 0, 0, 0
        tw, th = self.target_size

        scale = min(float(tw) / w, float(th) / h)
        nw, nh = int(w * scale), int(h * scale)

        left = (tw - nw) // 2
        top = (th - nh) // 2
        right = tw - nw - left
        bottom = th - nh - top

        im = cv2.resize(cv_image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if use_norm:
            im = (im - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
            im = im.astype(np.float32)

        im = np.ascontiguousarray(np.transpose(im, axes=(2, 0, 1)))
        im = np.expand_dims(im, axis=0).copy()

        return im, scale, (left, top)

    def detect(self, cv_image):
        """Perform object detection on an image"""
        h, w = cv_image.shape[:2]
        preprocessed_img, scale, (padx, pady) = self.preprocess(cv_image)

        # Run inference
        outputs = self.model.run(None, {self.input_name: preprocessed_img})
        bbox_preds, cls_scores, max_idxes = outputs

        num_proposals = cls_scores.shape[0]
        all_boxes = []
        all_scores = []
        all_classes = []

        for idx in range(num_proposals):
            max_idx = max_idxes[idx][0]
            score = cls_scores[idx][max_idx]

            if score < self.conf_thresh:
                continue

            x1, y1, x2, y2 = bbox_preds[idx][max_idx]

            x1 = (x1 - padx) / scale
            y1 = (y1 - pady) / scale
            x2 = (x2 - padx) / scale
            y2 = (y2 - pady) / scale

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(score)
            all_classes.append(max_idx)

        if not all_boxes:
            return []

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        keep = self.nms(all_boxes, all_scores, self.nms_thresh)

        detections = []
        for i in keep:
            x1, y1, x2, y2 = all_boxes[i]
            coco_cat_id = self.dataset.get_coco_category_id(int(all_classes[i]))
            if coco_cat_id == -1:
                continue

            detection = {
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'category_id': coco_cat_id,
                'score': float(all_scores[i])
            }
            detections.append(detection)

        return detections

def evaluate_on_coco(model_path, ann_file, img_dir, output_dir='detection_results'):
    """Evaluate object detection model on COCO dataset with grid search over thresholds"""
    os.makedirs(output_dir, exist_ok=True)

    # Define parameter ranges for grid search
    conf_thresholds = np.arange(0.2, 0.45, 0.05)  # 0.2 to 0.4 with step 0.05
    nms_thresholds = np.arange(0.4, 0.65, 0.05)  # 0.4 to 0.6 with step 0.05

    # Initialize dataset and detector
    dataset = COCODataset(ann_file, img_dir)
    detector = ObjectDetector(model_path, dataset)

    # Store results for each threshold combination
    grid_search_results = []

    # Grid search over thresholds
    for conf_thresh in conf_thresholds:
        for nms_thresh in nms_thresholds:
            logger.info(f"\nEvaluating with conf_thresh={conf_thresh:.2f}, nms_thresh={nms_thresh:.2f}")

            # Update detector thresholds
            detector.set_thresholds(conf_thresh, nms_thresh)

            results = []
            debug_info = {'total_images': 0, 'total_detections': 0, 'images_with_dets': 0}

            # Run inference on dataset
            for idx in tqdm(range(len(dataset))):
                image_path, img_info, img_id = dataset.get_image_info(idx)

                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Failed to load image: {image_path}")
                    continue

                debug_info['total_images'] += 1

                # Get detections
                detections = detector.detect(img)

                debug_info['total_detections'] += len(detections)
                if len(detections) > 0:
                    debug_info['images_with_dets'] += 1

                # Add image_id to detections
                for det in detections:
                    det['image_id'] = img_id
                    results.append(det)

            # Save results for this threshold combination
            result_file = os.path.join(output_dir,
                                       f'results_conf{conf_thresh:.2f}_nms{nms_thresh:.2f}.json')
            with open(result_file, 'w') as f:
                json.dump(results, f)

            # Evaluate results
            if len(results) > 0:
                coco_dt = dataset.coco.loadRes(results)
                cocoEval = COCOeval(dataset.coco, coco_dt, 'bbox')
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                mAP = cocoEval.stats[0]
            else:
                mAP = 0.0
                logger.warning("No detections found for this threshold combination!")

            # Store results
            grid_search_results.append({
                'conf_thresh': conf_thresh,
                'nms_thresh': nms_thresh,
                'mAP': mAP,
                'total_detections': debug_info['total_detections'],
                'images_with_dets': debug_info['images_with_dets']
            })

            logger.info(f"mAP (IoU=0.50:0.95): {mAP:.4f}")

    # Find best threshold combination
    best_result = max(grid_search_results, key=lambda x: x['mAP'])

    # Save grid search results to CSV
    csv_results = ['conf_thresh,nms_thresh,mAP,total_detections,images_with_dets']
    for result in grid_search_results:
        csv_results.append(f"{result['conf_thresh']:.2f},{result['nms_thresh']:.2f},"
                           f"{result['mAP']:.4f},{result['total_detections']},{result['images_with_dets']}")

    with open(os.path.join(output_dir, 'grid_search_results.csv'), 'w') as f:
        f.write('\n'.join(csv_results))

    logger.info("\nGrid Search Results:")
    logger.info(f"Best configuration: conf_thresh={best_result['conf_thresh']:.2f}, "
                f"nms_thresh={best_result['nms_thresh']:.2f}")
    logger.info(f"Best mAP: {best_result['mAP']:.4f}")

    return best_result['mAP']


if __name__ == "__main__":
    ann_file = '/home/pi/Downloads/instances_val2017_mini50.json'
    img_dir = '/home/pi/Downloads/val2017'
    model_path = '/home/pi/Downloads/final_fp16.rknn'
    mAP = evaluate_on_coco(model_path, ann_file, img_dir)
    print(f"\nBest mAP (IoU=0.50:0.95): {mAP:.4f}")