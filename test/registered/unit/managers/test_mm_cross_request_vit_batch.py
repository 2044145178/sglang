import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_utils
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class _DummyVisual:
    spatial_merge_unit = 4


class _DummyQwenLikeModel:
    def __init__(self):
        self.visual = _DummyVisual()
        self.batch_sizes = []

    def get_image_feature(self, items):
        self.batch_sizes.append(len(items))
        rows = []
        for item in items:
            if "image_grid_thw" in item.model_specific_data:
                length = int(torch.prod(item.image_grid_thw.reshape(-1, 3)[0]).item())
                length //= self.visual.spatial_merge_unit
            else:
                start, end = item.offsets[0]
                length = end - start + 1
            values = torch.arange(length, dtype=torch.float32).unsqueeze(1)
            rows.append(torch.cat([values, values + item.hash, values + item.hash], dim=1))
        return torch.cat(rows, dim=0)


class TestMmCrossRequestVitBatch(unittest.TestCase):
    def setUp(self):
        mm_utils.init_mm_embedding_cache(max_size=1024 * 1024)

    def _image_item(self, item_hash, offset, image_grid_thw=None):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=item_hash,
            offsets=[offset],
            feature=torch.zeros((1, 1)),
        )
        if image_grid_thw is not None:
            item.set("image_grid_thw", image_grid_thw)
        return item

    def test_batches_qwen_like_cache_misses_across_requests(self):
        model = _DummyQwenLikeModel()
        items = [
            self._image_item(11, (0, 1), torch.tensor([[1, 2, 4]])),
            self._image_item(23, (0, 1), torch.tensor([[1, 2, 4]])),
        ]

        with envs.SGLANG_MM_CROSS_REQUEST_VIT_BATCH.override(True):
            embedding, input_ids = mm_utils._get_chunked_prefill_embedding(
                data_embedding_func=model.get_image_feature,
                embedding_items=items,
                items_size=[0, 1, 2],
                prefix_length=[0, 0],
                extend_length=[2, 2],
                items_offset_list=[[(0, 1)], [(0, 1)]],
                input_ids=torch.tensor([0, 0, 0, 0]),
            )

        self.assertEqual(model.batch_sizes, [2])
        self.assertEqual(tuple(embedding.shape), (4, 3))
        self.assertEqual(input_ids.tolist(), [0, 0, 0, 0])
        self.assertTrue(torch.equal(embedding[:2], model.get_image_feature([items[0]])))
        self.assertTrue(torch.equal(embedding[2:], model.get_image_feature([items[1]])))

    def test_falls_back_when_true_image_length_is_not_known(self):
        model = _DummyQwenLikeModel()
        items = [
            self._image_item(11, (0, 1)),
            self._image_item(23, (0, 1)),
        ]

        with envs.SGLANG_MM_CROSS_REQUEST_VIT_BATCH.override(True):
            embedding, _ = mm_utils._get_chunked_prefill_embedding(
                data_embedding_func=model.get_image_feature,
                embedding_items=items,
                items_size=[0, 1, 2],
                prefix_length=[0, 0],
                extend_length=[2, 2],
                items_offset_list=[[(0, 1)], [(0, 1)]],
                input_ids=torch.tensor([0, 0, 0, 0]),
            )

        self.assertEqual(model.batch_sizes, [1, 1])
        self.assertEqual(tuple(embedding.shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
