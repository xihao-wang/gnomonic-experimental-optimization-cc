# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import json
import numpy as np
import torch
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .opts import opt

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=100, n_init=10,
                 temporal_model=None, temporal_alpha=1.0,
                 fuse_temporal_model=False, temporal_max_correction=0.02,
                 temporal_min_scale=0.02, temporal_veto_cost=0.8,
                 association_veto_model=None,
                 association_veto_threshold=0.7,
                 matching_debug_jsonl=None):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.temporal_model = temporal_model
        self.temporal_alpha = float(temporal_alpha)
        self.fuse_temporal_model = bool(fuse_temporal_model)
        self.temporal_max_correction = float(temporal_max_correction)
        self.temporal_min_scale = float(temporal_min_scale)
        self.temporal_veto_cost = float(temporal_veto_cost)
        self.association_veto_model = association_veto_model
        self.association_veto_threshold = float(association_veto_threshold)

        self.tracks = []
        self._next_id = 1
        self.last_ambiguous_tracks = []
        self.last_ambiguous_info = {}
        self.last_match_confidences = {}
        self.temporal_cost_override = None
        self.matching_debug_jsonl = matching_debug_jsonl
        self.matching_debug_frame = None
        self._matching_debug_call_index = 0

    def set_temporal_cost_override(self, track_indices, detection_indices, temporal_cost):
        self.temporal_cost_override = {
            "track_indices": list(track_indices),
            "detection_indices": list(detection_indices),
            "temporal_cost": np.asarray(temporal_cost, dtype=np.float32),
        }

    def clear_temporal_cost_override(self):
        self.temporal_cost_override = None

    @staticmethod
    def _cosine_distance_to_bank(feature, memory_bank):
        if len(memory_bank) == 0:
            return None
        feature = np.asarray(feature, dtype=np.float32)
        feature_norm = np.linalg.norm(feature)
        if feature_norm < 1e-12:
            return None
        feature = feature / feature_norm

        memory = np.asarray(memory_bank, dtype=np.float32)
        memory_norm = np.linalg.norm(memory, axis=1, keepdims=True)
        valid = memory_norm.reshape(-1) > 1e-12
        if not np.any(valid):
            return None
        memory = memory[valid] / memory_norm[valid]
        distances = 1.0 - np.dot(memory, feature)
        if opt.enable_topk_matching:
            k = min(opt.k, len(distances))
            return float(np.mean(np.sort(distances)[:k]))
        return float(np.min(distances))

    @staticmethod
    def _build_short_history(track, history_len):
        short_memory = getattr(track, "short_memory", [])
        if len(short_memory) > 0:
            valid_len = min(len(short_memory), history_len)
            items = [np.asarray(feat, dtype=np.float32) for feat in short_memory[-history_len:]][::-1]
            while len(items) < history_len:
                items.append(items[-1])
            return np.stack(items, axis=0), valid_len

        history = getattr(track, "det_feat_history", [])
        if len(history) > 0:
            valid_len = min(len(history), history_len)
            items = [np.asarray(feat, dtype=np.float32) for feat in history[-history_len:]][::-1]
            while len(items) < history_len:
                items.append(items[-1])
            return np.stack(items, axis=0), valid_len
        return None

    @staticmethod
    def _build_long_history(track, long_history_len):
        long_memory = getattr(track, "long_memory", [])
        if len(long_memory) > 0:
            valid_len = min(len(long_memory), long_history_len)
            items = [np.asarray(feat, dtype=np.float32) for feat in long_memory[-long_history_len:]]
            while len(items) < long_history_len:
                items.append(items[-1])
            return np.stack(items, axis=0), valid_len

        history = getattr(track, "det_feat_history", [])
        if len(history) > 0:
            valid_len = min(len(history), long_history_len)
            items = [np.asarray(feat, dtype=np.float32) for feat in history[-long_history_len:]]
            while len(items) < long_history_len:
                items.append(items[-1])
            return np.stack(items, axis=0), valid_len
        return None

    @staticmethod
    def _normalize_cost_rows(cost_matrix):
        normalized = cost_matrix.copy().astype(np.float32)
        big_cost = linear_assignment.INFTY_COST
        for row_idx in range(normalized.shape[0]):
            row = normalized[row_idx]
            valid_mask = row < big_cost
            if not np.any(valid_mask):
                continue
            valid = row[valid_mask]
            min_v = float(np.min(valid))
            max_v = float(np.max(valid))
            if max_v - min_v < 1e-12:
                normalized[row_idx, valid_mask] = 0.0
            else:
                normalized[row_idx, valid_mask] = (valid - min_v) / (max_v - min_v)
        return normalized

    def _temporal_cost_matrix(
            self, tracks, detections, track_indices, detection_indices,
            require_fusion=True):
        if require_fusion and not self.fuse_temporal_model:
            return None
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return None
        if self.temporal_cost_override is not None:
            override = self.temporal_cost_override
            source_tracks = override["track_indices"]
            source_dets = override["detection_indices"]
            source_cost = override["temporal_cost"]
            aligned = np.ones((len(track_indices), len(detection_indices)), dtype=np.float32)
            source_track_to_row = {track_idx: row for row, track_idx in enumerate(source_tracks)}
            source_det_to_col = {det_idx: col for col, det_idx in enumerate(source_dets)}
            for row, track_idx in enumerate(track_indices):
                source_row = source_track_to_row.get(track_idx)
                if source_row is None:
                    continue
                for col, detection_idx in enumerate(detection_indices):
                    source_col = source_det_to_col.get(detection_idx)
                    if source_col is not None:
                        aligned[row, col] = source_cost[source_row, source_col]
            return aligned
        if self.temporal_model is None:
            return None

        history_len = getattr(self.temporal_model, "history_len", 5)
        long_history_len = getattr(self.temporal_model, "long_history_len", 30)
        det_batch = []
        short_batch = []
        long_batch = []
        short_len_batch = []
        long_len_batch = []
        pair_indices = []
        temporal_cost = np.ones((len(track_indices), len(detection_indices)), dtype=np.float32)

        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            short_result = self._build_short_history(track, history_len)
            if short_result is None:
                continue
            short_hist, short_len = short_result
            long_result = self._build_long_history(track, long_history_len)
            if long_result is None:
                long_hist = np.repeat(short_hist[-1:, :], long_history_len, axis=0)
                long_len = 1
            else:
                long_hist, long_len = long_result
            for col, detection_idx in enumerate(detection_indices):
                det_feat = np.asarray(detections[detection_idx].feature, dtype=np.float32)
                norm = np.linalg.norm(det_feat)
                if norm > 1e-12:
                    det_feat = det_feat / norm
                det_batch.append(det_feat)
                short_batch.append(short_hist)
                long_batch.append(long_hist)
                short_len_batch.append(short_len)
                long_len_batch.append(long_len)
                pair_indices.append((row, col))

        if not pair_indices:
            return temporal_cost

        with torch.no_grad():
            temporal_device = next(self.temporal_model.parameters()).device
            logits = self.temporal_model(
                torch.from_numpy(np.stack(det_batch, axis=0)).to(temporal_device),
                torch.from_numpy(np.stack(short_batch, axis=0)).to(temporal_device),
                long_hist_feat=torch.from_numpy(np.stack(long_batch, axis=0)).to(temporal_device),
                short_hist_len=torch.from_numpy(np.asarray(short_len_batch, dtype=np.int64)).to(temporal_device),
                long_hist_len=torch.from_numpy(np.asarray(long_len_batch, dtype=np.int64)).to(temporal_device),
                return_attention=False,
            )
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        for idx, (row, col) in enumerate(pair_indices):
            temporal_cost[row, col] = 1.0 - float(probs[idx])
        return temporal_cost

    def _fuse_temporal_cost(self, base_cost, temporal_cost):
        if temporal_cost is None:
            return base_cost
        gamma = max(self.temporal_alpha, 0.0)
        fused = base_cost.copy()
        big_cost = linear_assignment.INFTY_COST
        min_scale = max(self.temporal_min_scale, 0.0)
        max_correction = max(self.temporal_max_correction, 0.0)
        eps = 1e-12

        for row_idx in range(base_cost.shape[0]):
            valid_mask = base_cost[row_idx] < big_cost
            if np.sum(valid_mask) < 2:
                continue
            learned_row = temporal_cost[row_idx, valid_mask].astype(np.float32)
            base_row = base_cost[row_idx, valid_mask].astype(np.float32)
            learned_delta = learned_row - float(np.mean(learned_row))
            learned_scale = float(np.std(learned_delta))
            if learned_scale < eps:
                continue
            base_scale = float(np.std(base_row))
            effective_scale = max(base_scale, min_scale)
            beta = gamma * effective_scale / max(learned_scale, eps)
            correction = beta * learned_delta
            correction = np.clip(correction, -max_correction, max_correction)
            fused[row_idx, valid_mask] = base_row + correction

        # Row-wise correction handles "one track chooses among detections".
        # Column-wise correction handles "multiple tracks compete for one detection".
        for col_idx in range(base_cost.shape[1]):
            valid_mask = base_cost[:, col_idx] < big_cost
            if np.sum(valid_mask) < 2:
                continue
            learned_col = temporal_cost[valid_mask, col_idx].astype(np.float32)
            base_col = base_cost[valid_mask, col_idx].astype(np.float32)
            learned_delta = learned_col - float(np.mean(learned_col))
            learned_scale = float(np.std(learned_delta))
            if learned_scale < eps:
                continue
            base_scale = float(np.std(base_col))
            effective_scale = max(base_scale, min_scale)
            beta = gamma * effective_scale / max(learned_scale, eps)
            correction = beta * learned_delta
            correction = np.clip(correction, -max_correction, max_correction)
            fused[valid_mask, col_idx] = fused[valid_mask, col_idx] + correction

        valid_mask = fused < big_cost
        if np.any(valid_mask):
            min_value = float(np.min(fused[valid_mask]))
            if min_value < 0.0:
                fused[valid_mask] = fused[valid_mask] - min_value
        return fused

    def _apply_temporal_veto(self, cost_matrix, temporal_cost):
        if self.temporal_veto_cost < 0:
            return cost_matrix
        if temporal_cost is None:
            return cost_matrix
        if temporal_cost.shape != cost_matrix.shape:
            return cost_matrix
        vetoed = cost_matrix.copy()
        # temporal_cost is initialized to 1.0 for pairs not evaluated by the
        # learned model. Avoid vetoing those default placeholders.
        evaluated_mask = temporal_cost < 0.999999
        veto_mask = evaluated_mask & (temporal_cost > self.temporal_veto_cost)
        vetoed[veto_mask] = linear_assignment.INFTY_COST
        return vetoed

    def _gate_cost_matrix_with_temporal_rescue(
            self, cost_matrix, temporal_cost, tracks, detections,
            track_indices, detection_indices, gated_cost=linear_assignment.INFTY_COST):
        assert not opt.MC
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return cost_matrix

        gating_threshold = kalman_filter.chi2inv95[4]
        rescue_gating_multiplier = 2.0
        temperature = 0.5
        measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])

        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            gating_distance = track.kf.gating_distance(
                track.mean, track.covariance, measurements, only_position=False
            )
            gated_mask = gating_distance > gating_threshold
            if not np.any(gated_mask):
                continue

            if temporal_cost is None or temporal_cost.shape != cost_matrix.shape:
                cost_matrix[row, gated_mask] = gated_cost
                continue

            temporal_row = temporal_cost[row].astype(np.float32)
            valid_temporal = np.isfinite(temporal_row)
            if np.sum(valid_temporal) < 2:
                cost_matrix[row, gated_mask] = gated_cost
                continue

            logits = -temporal_row[valid_temporal]
            logits = logits / max(temperature, 1e-12)
            logits = logits - float(np.max(logits))
            probs = np.exp(logits)
            probs = probs / max(float(np.sum(probs)), 1e-12)
            valid_cols = np.flatnonzero(valid_temporal)
            best_local = int(np.argmax(probs))
            best_col = int(valid_cols[best_local])
            prominence = float(np.mean(probs) + np.std(probs))

            for col, is_gated in enumerate(gated_mask):
                if not is_gated:
                    continue
                rescue = (
                    col == best_col
                    and float(probs[best_local]) >= prominence
                    and float(gating_distance[col]) <= gating_threshold * rescue_gating_multiplier
                )
                if not rescue:
                    cost_matrix[row, col] = gated_cost
        return cost_matrix

    def _kalman_gating_distance_matrix(
            self, tracks, detections, track_indices, detection_indices):
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)
        measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
        distances = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)
        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            distances[row] = track.kf.gating_distance(
                track.mean, track.covariance, measurements, only_position=False
            )
        return distances

    @staticmethod
    def _tlwh_iou(a, b):
        ax, ay, aw, ah = [float(v) for v in a]
        bx, by, bw, bh = [float(v) for v in b]
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        if union <= 0:
            return 0.0
        return float(inter / union)

    def _iou_matrix(self, tracks, detections, track_indices, detection_indices):
        out = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)
        for row, track_idx in enumerate(track_indices):
            track_box = tracks[track_idx].to_tlwh()
            for col, detection_idx in enumerate(detection_indices):
                out[row, col] = self._tlwh_iou(track_box, detections[detection_idx].tlwh)
        return out

    def _apply_association_veto(
            self, cost_matrix, tracks, detections, track_indices, detection_indices,
            base_cost, temporal_cost, gating_distance, iou_matrix):
        if self.association_veto_model is None or self.association_veto_threshold < 0:
            return cost_matrix
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return cost_matrix

        features = []
        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            for col, _detection_idx in enumerate(detection_indices):
                temporal = 1.0 - float(temporal_cost[row, col]) if temporal_cost is not None else 0.0
                if temporal_cost is not None and float(temporal_cost[row, col]) >= 0.999999:
                    temporal = 0.0
                features.append([
                    temporal,
                    float(base_cost[row, col]),
                    float(gating_distance[row, col]),
                    float(iou_matrix[row, col]),
                    float(track.time_since_update),
                ])

        probs = self.association_veto_model.predict_proba(np.asarray(features, dtype=np.float32))
        probs = np.asarray(probs, dtype=np.float32).reshape(len(track_indices), len(detection_indices))
        vetoed = cost_matrix.copy()
        vetoed[probs < self.association_veto_threshold] = linear_assignment.INFTY_COST
        return vetoed

    @staticmethod
    def _matrix_to_list(matrix):
        if matrix is None:
            return None
        arr = np.asarray(matrix, dtype=np.float64)
        out = []
        for row in arr:
            out.append([
                None if not np.isfinite(value) else float(value)
                for value in row
            ])
        return out

    def _write_matching_debug(
            self, stage, tracks, detections, track_indices, detection_indices,
            base_cost, temporal_cost, fused_cost_before_kalman,
            gating_distance, final_cost_after_kalman):
        if not self.matching_debug_jsonl:
            return

        big_cost = linear_assignment.INFTY_COST
        gating_threshold = kalman_filter.chi2inv95[4]
        row_summary = []
        assigned_by_final_cost = {}
        if final_cost_after_kalman.size > 0:
            assignment = linear_assignment.linear_assignment(final_cost_after_kalman.copy())
            for row, col in assignment:
                if float(final_cost_after_kalman[row, col]) < big_cost:
                    assigned_by_final_cost[int(row)] = int(col)

        for row, track_idx in enumerate(track_indices):
            learned_best_col = None
            if temporal_cost is not None and temporal_cost.shape == final_cost_after_kalman.shape:
                temporal_row = temporal_cost[row]
                valid_temporal = np.isfinite(temporal_row)
                if np.any(valid_temporal):
                    learned_best_col = int(np.argmin(np.where(
                        valid_temporal, temporal_row, np.inf
                    )))

            final_best_col = None
            finite_final = final_cost_after_kalman[row] < big_cost
            if np.any(finite_final):
                final_best_col = int(np.argmin(np.where(
                    finite_final, final_cost_after_kalman[row], np.inf
                )))

            assigned_col = assigned_by_final_cost.get(row)
            learned_best_gated = None
            learned_best_gate_distance = None
            learned_best_final_cost = None
            if learned_best_col is not None:
                learned_best_final_cost = float(final_cost_after_kalman[row, learned_best_col])
                learned_best_gated = bool(learned_best_final_cost >= big_cost)
                learned_best_gate_distance = float(gating_distance[row, learned_best_col])

            row_summary.append({
                "track_index": int(track_idx),
                "track_id": int(tracks[track_idx].track_id),
                "learned_best_detection_index": (
                    None if learned_best_col is None
                    else int(detection_indices[learned_best_col])
                ),
                "learned_best_temporal_cost": (
                    None if learned_best_col is None or temporal_cost is None
                    else float(temporal_cost[row, learned_best_col])
                ),
                "learned_best_gating_distance": learned_best_gate_distance,
                "learned_best_over_gate_threshold": (
                    None if learned_best_gate_distance is None
                    else bool(learned_best_gate_distance > gating_threshold)
                ),
                "learned_best_final_cost": learned_best_final_cost,
                "learned_best_gated_to_infty": learned_best_gated,
                "final_best_detection_index": (
                    None if final_best_col is None
                    else int(detection_indices[final_best_col])
                ),
                "final_best_cost": (
                    None if final_best_col is None
                    else float(final_cost_after_kalman[row, final_best_col])
                ),
                "assigned_by_final_cost_detection_index": (
                    None if assigned_col is None
                    else int(detection_indices[assigned_col])
                ),
                "learned_best_equals_final_best": (
                    None if learned_best_col is None or final_best_col is None
                    else bool(learned_best_col == final_best_col)
                ),
                "learned_best_equals_assigned_by_final_cost": (
                    None if learned_best_col is None or assigned_col is None
                    else bool(learned_best_col == assigned_col)
                ),
            })

        record = {
            "frame": None if self.matching_debug_frame is None else int(self.matching_debug_frame),
            "stage": stage,
            "call_index": int(self._matching_debug_call_index),
            "track_indices": [int(i) for i in track_indices],
            "track_ids": [int(tracks[i].track_id) for i in track_indices],
            "track_ages": [int(tracks[i].time_since_update) for i in track_indices],
            "track_tlwhs": [
                [float(v) for v in tracks[i].to_tlwh()]
                for i in track_indices
            ],
            "detection_indices": [int(i) for i in detection_indices],
            "detection_confidences": [float(detections[i].confidence) for i in detection_indices],
            "detection_tlwhs": [
                [float(v) for v in detections[i].tlwh]
                for i in detection_indices
            ],
            "base_cost": self._matrix_to_list(base_cost),
            "temporal_cost": self._matrix_to_list(temporal_cost),
            "fused_cost_before_kalman": self._matrix_to_list(fused_cost_before_kalman),
            "gating_distance": self._matrix_to_list(gating_distance),
            "iou": self._matrix_to_list(self._iou_matrix(tracks, detections, track_indices, detection_indices)),
            "final_cost_after_kalman": self._matrix_to_list(final_cost_after_kalman),
            "gating_threshold": float(gating_threshold),
            "infty_cost": float(big_cost),
            "row_summary": row_summary,
        }
        self._matching_debug_call_index += 1
        with open(self.matching_debug_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_assignment_debug(self, stage, matches):
        if not self.matching_debug_jsonl:
            return
        record = {
            "frame": None if self.matching_debug_frame is None else int(self.matching_debug_frame),
            "stage": stage,
            "call_index": int(self._matching_debug_call_index),
            "assignments": [
                {
                    "track_index": int(track_idx),
                    "track_id": int(self.tracks[track_idx].track_id),
                    "detection_index": int(detection_idx),
                }
                for track_idx, detection_idx in matches
            ],
        }
        self._matching_debug_call_index += 1
        with open(self.matching_debug_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _cost_to_confidence(cost, max_cost):
        if max_cost <= 0:
            return 0.0
        value = 1.0 - float(cost) / float(max_cost)
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _assignment_cost(cost_matrix):
        if cost_matrix.size == 0:
            return None
        big_cost = linear_assignment.INFTY_COST
        indices = linear_assignment.linear_assignment(cost_matrix.copy())
        if len(indices) == 0:
            return None
        total = 0.0
        for row, col in indices:
            value = float(cost_matrix[row, col])
            if value >= big_cost:
                return None
            total += value
        return total

    def _combo_match_confidences(self, matches, cost_lookup, max_cost):
        if not matches:
            return {}

        matched_tracks = [track_idx for track_idx, _ in matches]
        matched_detections = [detection_idx for _, detection_idx in matches]
        detection_candidates = list(matched_detections)
        for track_idx in matched_tracks:
            for lookup_track_idx, detection_idx in cost_lookup.keys():
                if lookup_track_idx == track_idx and detection_idx not in detection_candidates:
                    detection_candidates.append(detection_idx)
        big_cost = linear_assignment.INFTY_COST
        cost_matrix = np.full(
            (len(matched_tracks), len(detection_candidates)),
            big_cost,
            dtype=np.float32,
        )

        for row, track_idx in enumerate(matched_tracks):
            for col, detection_idx in enumerate(detection_candidates):
                cost_matrix[row, col] = float(
                    cost_lookup.get((track_idx, detection_idx), big_cost)
                )

        current_total = 0.0
        for row, (track_idx, detection_idx) in enumerate(matches):
            col = detection_candidates.index(detection_idx)
            cost = float(cost_matrix[row, col])
            if cost >= big_cost:
                return {
                    pair: self._cost_to_confidence(
                        cost_lookup.get(pair, max_cost), max_cost
                    )
                    for pair in matches
                }
            current_total += cost

        margin_scale = max(float(getattr(opt, "match_conf_margin_scale", 0.02)), 1e-12)
        confidences = {}
        for row, (track_idx, detection_idx) in enumerate(matches):
            col = detection_candidates.index(detection_idx)
            matched_cost = float(cost_matrix[row, col])
            alternative_matrix = cost_matrix.copy()
            alternative_matrix[row, col] = big_cost
            alternative_total = self._assignment_cost(alternative_matrix)

            absolute_conf = self._cost_to_confidence(matched_cost, max_cost)
            if alternative_total is None:
                combo_conf = 1.0
            else:
                combo_margin = float(alternative_total - current_total)
                combo_conf = float(np.clip(combo_margin / margin_scale, 0.0, 1.0))
            confidences[(track_idx, detection_idx)] = absolute_conf * combo_conf
        return confidences

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def camera_update(self, video, frame):
        for track in self.tracks:
            track.camera_update(video, frame)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[tracker_pipeline.strongsort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            track = self.tracks[track_idx]
            track.update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_missed()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            if not opt.EMA:
                track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        self.last_match_confidences = {}
        for track in self.tracks:
            track.match_confidence = None
        match_costs = {}
        metric_stage = "unknown"

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            if opt.enable_memory_matching:
                candidate_tracks = [tracks[i] for i in track_indices]
                cost_matrix = self.metric.distance_with_memory(features, candidate_tracks)
            else:
                targets = np.array([tracks[i].track_id for i in track_indices])
                cost_matrix = self.metric.distance(features, targets)
            base_cost = cost_matrix.copy()
            temporal_cost = self._temporal_cost_matrix(
                tracks, dets, track_indices, detection_indices
            )
            cost_matrix = self._fuse_temporal_cost(cost_matrix, temporal_cost)
            if (
                    self.fuse_temporal_model
                    and temporal_cost is not None
                    and self.temporal_veto_cost >= 0):
                cost_matrix = self._apply_temporal_veto(cost_matrix, temporal_cost)
            fused_cost_before_kalman = cost_matrix.copy()
            gating_distance = self._kalman_gating_distance_matrix(
                tracks, dets, track_indices, detection_indices
            )
            iou_matrix = self._iou_matrix(tracks, dets, track_indices, detection_indices)
            association_veto_applied = (
                self.association_veto_model is not None
                and self.association_veto_threshold >= 0
            )
            cost_matrix = self._apply_association_veto(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                base_cost,
                temporal_cost,
                gating_distance,
                iou_matrix,
            )
            if self.fuse_temporal_model and temporal_cost is not None and not opt.MC and not association_veto_applied:
                cost_matrix = self._gate_cost_matrix_with_temporal_rescue(
                    cost_matrix, temporal_cost, tracks, dets,
                    track_indices, detection_indices)
            else:
                cost_matrix = linear_assignment.gate_cost_matrix(
                    cost_matrix, tracks, dets, track_indices,
                    detection_indices)
            self._write_matching_debug(
                f"appearance_metric_{metric_stage}",
                tracks,
                dets,
                track_indices,
                detection_indices,
                base_cost,
                temporal_cost,
                fused_cost_before_kalman,
                gating_distance,
                cost_matrix,
            )
            for row, track_idx in enumerate(track_indices):
                for col, detection_idx in enumerate(detection_indices):
                    match_costs[(track_idx, detection_idx)] = float(cost_matrix[row, col])

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Detect split ambiguity before the standard assignment step.
        detection_indices = list(range(len(detections)))
        ambiguous_tracks = []
        ambiguous_info = {}
        metric_stage = "pre_ambiguity"
        if confirmed_tracks and detection_indices:
            cost_matrix = gated_metric(
                self.tracks, detections, confirmed_tracks, detection_indices
            )
            big_cost = 1e5
            for row_idx, track_idx in enumerate(confirmed_tracks):
                row = cost_matrix[row_idx]
                valid = []
                for col_idx, dist in enumerate(row):
                    if dist < big_cost:
                        valid.append((detection_indices[col_idx], float(dist)))

                if len(valid) < 2:
                    continue

                valid.sort(key=lambda x: x[1])
                det_a, d1 = valid[0]
                det_b, d2 = valid[1]

                if (
                    d1 < opt.ambiguity_distance_threshold
                    and d2 < opt.ambiguity_distance_threshold
                    and abs(d1 - d2) < opt.ambiguity_margin
                ):
                    ambiguous_tracks.append(self.tracks[track_idx].track_id)
                    ambiguous_info[self.tracks[track_idx].track_id] = {
                        "candidates": [det_a, det_b],
                        "distances": [d1, d2],
                    }

        self.last_ambiguous_tracks = ambiguous_tracks
        self.last_ambiguous_info = ambiguous_info

        # Associate confirmed tracks using appearance features.
        metric_stage = "cascade"
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        self._write_assignment_debug("appearance_assignments", matches_a)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        self._write_assignment_debug("iou_assignments", matches_b)

        matches = matches_a + matches_b
        appearance_confidences = self._combo_match_confidences(
            matches_a, match_costs, self.metric.matching_threshold
        )
        for track_idx, detection_idx in matches_a:
            confidence = appearance_confidences.get((track_idx, detection_idx), 0.0)
            self.tracks[track_idx].match_confidence = confidence
            self.last_match_confidences[(track_idx, detection_idx)] = confidence
        if matches_b:
            iou_costs = {}
            iou_track_indices = [track_idx for track_idx, _ in matches_b]
            iou_detection_indices = [detection_idx for _, detection_idx in matches_b]
            if iou_track_indices and iou_detection_indices:
                iou_matrix = iou_matching.iou_cost(
                    self.tracks,
                    detections,
                    iou_track_indices,
                    iou_detection_indices,
                )
                for row, track_idx in enumerate(iou_track_indices):
                    for col, detection_idx in enumerate(iou_detection_indices):
                        iou_costs[(track_idx, detection_idx)] = float(iou_matrix[row, col])
            iou_confidences = self._combo_match_confidences(
                matches_b, iou_costs, self.max_iou_distance
            )
            for track_idx, detection_idx in matches_b:
                confidence = iou_confidences.get((track_idx, detection_idx), 0.0)
                self.tracks[track_idx].match_confidence = confidence
                self.last_match_confidences[(track_idx, detection_idx)] = confidence
        unmatched_tracks = list(set(
            unmatched_tracks_a + unmatched_tracks_b
        ))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, self.n_init, self.max_age,
            detection.feature, detection.confidence))
        self._next_id += 1
