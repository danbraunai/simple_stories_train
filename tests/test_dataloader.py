import torch

from simple_stories_train.dataloaders import DatasetConfig, create_data_loader


def test_dataloader_with_ddp():
    # Load a small dataset from the datasets library
    dataset_name = "lennart-finke/SimpleStories"
    split = "train"

    context_width = 128
    dataset_config = DatasetConfig(
        dataset_name=dataset_name,
        is_tokenized=False,
        tokenizer_file_path="simple_stories_train/tokenizer/stories-3072.json",
        streaming=True,
        split=split,
        n_ctx=context_width,
        seed=42,
        column_name="story",
        ddp_rank=0,
        ddp_world_size=1,
    )

    batch_size = 2
    buffer_size = 1000
    global_seed = 0

    # Create two DataLoaders with the same parameters and seed, to see they give the same data
    loader1, _ = create_data_loader(dataset_config, batch_size, buffer_size, global_seed)
    loader2, _ = create_data_loader(dataset_config, batch_size, buffer_size, global_seed)

    # Compare the first few batches
    num_batches_to_check = 5
    for i, (batch1, batch2) in enumerate(zip(loader1, loader2)):
        assert(batch1['input_ids'].shape == (batch_size, context_width))
        assert torch.equal(batch1['input_ids'], batch2['input_ids']), f"Batch {i} is not the same"
        if i >= num_batches_to_check - 1:
            break

    # Now test that data is different when ddp_rank is different
    dataset_config_rank0 = DatasetConfig(
        dataset_name=dataset_name,
        is_tokenized=False,
        tokenizer_file_path=dataset_config.tokenizer_file_path,
        streaming=True,
        split=split,
        n_ctx=context_width,
        seed=42,
        column_name="story",
        ddp_rank=0,
        ddp_world_size=2,
    )

    dataset_config_rank1 = DatasetConfig(
        dataset_name=dataset_name,
        is_tokenized=False,
        tokenizer_file_path=dataset_config.tokenizer_file_path,
        streaming=True,
        split=split,
        n_ctx=context_width,
        seed=42,
        column_name="story",
        ddp_rank=1,
        ddp_world_size=2,
    )

    # Create DataLoaders for rank 0 and rank 1
    loader_rank0, _ = create_data_loader(dataset_config_rank0, batch_size, buffer_size, global_seed)
    loader_rank1, _ = create_data_loader(dataset_config_rank1, batch_size, buffer_size, global_seed)

    # Compare the first few batches to ensure they are different
    batches_are_same = True
    for i, (batch0, batch1) in enumerate(zip(loader_rank0, loader_rank1)):
        if not torch.equal(batch0['input_ids'], batch1['input_ids']):
            batches_are_same = False
            break
        if i >= num_batches_to_check - 1:
            break

    assert not batches_are_same, "Batches are the same for different ddp_ranks"
