filter_rule_base_pipeline = [
    dict(type="ImageSizeFilter", min_size=14),
    dict(type="ImageAspectRatioFilter", max_aspect_ratio=200),
    dict(type="ImageExtFilter"),
    dict(type="ConvLengthFilter", min_length=1, max_length=300),
    dict(type="LengthAnomalyFilter",
         min_length=2,
         max_length=4096,
         check_question=True,
         check_answer=False,
         check_ori_answer=False,
         use_tokenizer=True),
    dict(type="TextRepeatFilter",
         check_question=True,
         check_answer=False,
         check_ori_answer=False)
]

filter_rule_base_for_question = dict(
    cfg=dict(type="SubPipeline", operators=filter_rule_base_pipeline),
    priority=0,
)

pipeline = dict(type="Pipeline",
                operations={'filter_rule_base_for_question': filter_rule_base_for_question})
