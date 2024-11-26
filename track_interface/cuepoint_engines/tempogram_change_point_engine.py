from typing import List

import librosa
import ruptures as rpt

from track_interface.cuepoint_engines.changepoint_engine import ChangePointEngine

HOP_LENGTH = 1024


class TempogramChangePointEngine(ChangePointEngine):
    def _get_tempogram(self):
        y, sr = librosa.load(self.file_path, sr=None)
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

        return librosa.feature.tempogram(
            onset_envelope=oenv,
            sr=sr,
            hop_length=HOP_LENGTH,
        ), sr

    def _get_sum_of_costs(self, algo, n_bkps):
        bkps = algo.predict(n_bkps=n_bkps)
        return algo.cost.sum_of_costs(bkps)

    def _change_point_detection(self, tempogram, sr):
        model = rpt.KernelCPD(kernel="linear").fit(tempogram.T)
        n_bkps_max = 16

        _ = model.predict(n_bkps_max)
        costs = [
            self._get_sum_of_costs(model, bkps) for bkps in range(1, n_bkps_max + 1)
        ]
        largest_diff_idx = largest_diff = 0
        for idx, cost in enumerate(costs):
            if idx < 2:
                continue
            diff = costs[idx - 2] + cost
            if diff > largest_diff:
                largest_diff_idx, largest_diff = idx - 1, diff

        opt_num_bkps = largest_diff_idx + 1
        bkps = model.predict(n_bkps=opt_num_bkps)
        return librosa.frames_to_time(bkps, sr=sr, hop_length=HOP_LENGTH)

    def generate_cuepoints(self) -> List[int]:
        tempogram, sr = self._get_tempogram()
        change_points = self._change_point_detection(tempogram, sr)
        first_beat_change_points = self._convert_changepoints_to_first_beats(
            change_points
        )
        processed_change_points = self._post_process(first_beat_change_points)
        return processed_change_points
