import os
import numpy as np
import torch
from mmcv.cnn import fuse_conv_bn
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

config = "/root/workspaces/iTPN/det/configs/itpn/clip_itpn_base_3x_ld090_dp030.py"
checkpoint = "/root/workspaces/iTPN/det/work_dirs/clip_itpn_base_3x_ld090_dp030/latest.pth"
img_path = "/root/workspaces/iTPN/det/demo/demo.jpg"
model = init_detector(config, checkpoint, device="cpu")
model = fuse_conv_bn(model)

class MaskRCNNExporter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.num_levels = 5
        self.featmap_sizes = [(120, 120), (60, 60), (30, 30), (15, 15), (7, 7)]
        self.mlvl_anchors = self.model.rpn_head.prior_generator.grid_priors(self.featmap_sizes, dtype=torch.float32,
                                                                            device="cpu")
        # self.mlvl_anchors = torch.cat(self.mlvl_anchors, dim=0).view(1, -1, 4)
        self.nms_pre = 120
        num_classes = 80
        self.labels = torch.arange(num_classes, dtype=torch.long, device="cpu")

    def forward(self, x):
        bs = int(x.shape[0])
        feats = self.model.extract_feat(x)
        cls_scores, bbox_preds = self.model.rpn_head(feats)

        # remove batch
        cls_score_list = select_single_mlvl(cls_scores, 0)  # [CHW, ...]
        bbox_pred_list = select_single_mlvl(bbox_preds, 0)

        bs = int(x.shape[0])
        # get_bboxes_single
        # level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for level_idx in range(5):
            rpn_cls_score = cls_score_list[level_idx]  # CHW
            rpn_bbox_pred = bbox_pred_list[level_idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0).reshape(-1)  # HWC
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            scores = rpn_cls_score.sigmoid()

            anchors = self.mlvl_anchors[level_idx]

            ranked_scores, rank_inds = scores.sort(descending=True)
            topk_inds = rank_inds[:self.nms_pre]
            scores = ranked_scores[:self.nms_pre]
            rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
            anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            # level_ids.append(scores.new_full((scores.size(0), ), level_idx, dtype=torch.long))

        # bbox_post_process
        # min_bbox_size = 0
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.model.rpn_head.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=None)
        scores, inds = scores.sort(descending=True)
        proposals = proposals[inds]
        proposals = torch.cat([proposals, scores[:, None]], dim=-1)
        batch_id = 0
        img_inds = proposals.new_full((proposals.size(0), 1), batch_id)
        rois = torch.cat([img_inds, proposals[:, :4]], dim=-1)
        bbox_results = self.model.roi_head._bbox_forward(feats, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        det_bbox, det_label = self.model.roi_head.bbox_head.get_bboxes(rois, cls_score, bbox_pred, None, None)
        bboxes = det_bbox.view(det_label.size(0), -1, 4)
        scores = det_label[:, :-1]  # num_proposals, 80
        max_idxes = torch.argmax(scores, dim=-1, keepdim=True)
        return bboxes, scores, max_idxes

    def forward0(self, images):
        bs = int(x.shape[0])
        feats = self.model.extract_feat(images)
        cls_scores, bbox_preds = self.model.rpn_head(feats)
        cls_scores_new = []
        bbox_preds_new = []
        for level_idx in range(5):
            rpn_cls_score = cls_scores[level_idx]  # NCHW
            rpn_bbox_pred = bbox_preds[level_idx]
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).reshape(bs, -1)
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).reshape(bs, -1, 4)
            rpn_cls_score = rpn_cls_score.sigmoid()
            cls_scores_new.append(rpn_cls_score)
            bbox_preds_new.append(rpn_bbox_pred)
            print(rpn_bbox_pred.shape)
        cls_scores = torch.cat(cls_scores_new, dim=1)
        bbox_preds = torch.cat(bbox_preds_new, dim=1)
        bbox_preds = self.model.rpn_head.bbox_coder.decode(self.mlvl_anchors, bbox_preds, max_shape=None)
        num_proposals = int(bbox_preds.shape[1])
        batch_indices = torch.arange(bs).view(-1, 1, 1).expand(-1, num_proposals, 1)
        rois = torch.cat([batch_indices, bbox_preds], dim=2)
        bbox_results = self.model.roi_head._bbox_forward(feats, rois[0])
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        det_bbox, det_label = self.model.roi_head.bbox_head.get_bboxes(rois[0], cls_score, bbox_pred, None, None)
        bboxes = det_bbox.view(det_label.size(0), -1, 4)
        scores = det_label[:, :-1]  # num_proposals, 80
        max_idxes = torch.argmax(scores, dim=-1, keepdim=True)
        return bboxes, scores, max_idxes


m = MaskRCNNExporter()
m.cpu()
m.eval()

x = torch.from_numpy(np.load("images.npy")).cpu()
torch.onnx.export(m, x, "final.onnx", verbose=True, opset_version=13, input_names=["images"],
                  output_names=["bboxes", "scores", "max_idxes"])
