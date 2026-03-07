dataloader = dict(
    type="StandardDataLoader",
    data_root="./",
    dataset="",
    batch_size=512,
    num_workers=16,
    cache_dir=None,
    resize_image=True,
    resize_image_size=2048,
)

datasaver = dict(
    type="StandardDataSaver",
    dataset="",
    output_dir="",
    save_yaml_config=True,
    save_yaml_name="",
)
