class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
            mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(possibly_batched_index[0])

        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]

        if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
            return self.collate_fn(data), mix_index.cpu(), mix_lam, mix_bbox, soft_label.cpu()
        else:
            return self.collate_fn(data)