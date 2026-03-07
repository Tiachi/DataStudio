rewriter_rule_base_pipeline = [
]

rewriter_rule_base = dict(
    cfg=dict(type="SubPipeline", operators=rewriter_rule_base_pipeline),
    priority=1,
)

pipeline = dict(type="Pipeline",
                operations={'rewriter_rule_base': rewriter_rule_base})
