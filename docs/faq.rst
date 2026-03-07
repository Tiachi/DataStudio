Frequently Asked Questions
===========================

This page answers common questions about DataStudio.

.. contents:: Table of Contents
   :local:
   :depth: 2

General Questions
-----------------

What is DataStudio?
~~~~~~~~~~~~~~~~~~~

DataStudio is an industrial-grade multimodal data processing pipeline designed for preparing training data for Multimodal Large Language Models (MLLMs). It's the data curation engine behind the `Bee project <https://open-bee.github.io/>`_ and the `Honey-Data-15M <https://huggingface.co/datasets/Open-Bee/Honey-Data-15M>`_ dataset.

What makes DataStudio different from other data processing tools?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataStudio is specifically designed for multimodal (image + text) data with:

- **Native MLLM integration**: Use vision-language models for intelligent filtering and rewriting
- **Massive parallelism**: Multi-process async concurrent API requests (8192+ via ``MPOpenAIAPI``)
- **LMDB image caching**: Blazing-fast image I/O with sharded caching
- **Config-driven pipelines**: Reproducible, shareable processing workflows

Is DataStudio free to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! DataStudio is open-source under the Apache License 2.0. You can use it freely for both research and commercial purposes.

Installation & Setup
--------------------

What Python version is required?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataStudio requires **Python 3.10 or higher**.

.. code-block:: bash

   python --version  # Should be 3.10+

How do I install DataStudio?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/Open-Bee/DataStudio.git
   cd DataStudio
   pip install -r requirements.txt
   pip install -e .

Do I need a GPU?
~~~~~~~~~~~~~~~~

**No GPU is required** for running DataStudio itself. However:

- If you're using locally deployed MLLMs (e.g., vLLM, SGLang), you'll need GPUs
- If you're using cloud APIs (OpenAI, etc.), no GPU is needed

How much disk space do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It depends on your dataset:

- **LMDB cache**: Roughly 50-100% of your original image size
- **Output data**: Similar to input data size
- **Working space**: ~10GB for checkpoints and logs

We recommend having **at least 2x your dataset size** in free disk space.

Data Format
-----------

What data formats does DataStudio support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataStudio supports:

- **JSON**: Single file with array of samples
- **JSONL**: One sample per line (recommended for large datasets)

What should my data look like?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
       "id": "unique_sample_id",
       "image": "path/to/image.jpg",
       "conversations": [
           {"from": "human", "value": "What is in this image?"},
           {"from": "gpt", "value": "This is a cat."}
       ]
   }

Can I process multi-image samples?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Use a list for the ``image`` field:

.. code-block:: json

   {
       "id": "multi_image_sample",
       "image": ["image1.jpg", "image2.jpg", "image3.jpg"],
       "conversations": [
           {"from": "human", "value": "Compare these three images."},
           {"from": "gpt", "value": "The first image shows..."}
       ]
   }

Can I process text-only data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, simply omit the ``image`` field or set ``use_image=False`` in your config:

.. code-block:: python

   dataloader = dict(
       use_image=False,
       # ...
   )

Pipeline & Operators
--------------------

How do I choose which operators to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with these common patterns:

1. **Basic cleaning**: ``ConvLengthFilter`` + ``RemoveThinkRewriter``
2. **Quality filtering**: Add ``MLLMFilter`` with a quality prompt
3. **Content enhancement**: Add ``MLLMRewriter`` for CoT enrichment

What's the difference between Filter and Rewriter?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Filter**: Decides whether to keep or remove a sample (binary decision)
- **Rewriter**: Modifies the content of a sample (transformation)

Can I combine multiple operators?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! That's the core design of DataStudio:

.. code-block:: python

   pipeline = Pipeline([
       Filter1(),
       Filter2(),
       Rewriter1(),
       Rewriter2(),
   ])

Operators execute in order. Filtered samples are removed before reaching subsequent operators.

How do I create custom operators?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the :doc:`guide/development` guide for detailed instructions.

MLLM Integration
----------------

Which MLLM providers are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataStudio supports any **OpenAI-compatible API**, including:

- OpenAI (GPT-4o, GPT-4V)
- Anthropic (Claude)
- Local deployments (vLLM, SGLang, Ollama)
- Cloud services (Azure OpenAI, AWS Bedrock)

How do I use a local model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deploy your model with vLLM or SGLang, then configure DataStudio:

.. code-block:: python

   model = dict(
       model="Qwen3-VL-30B-A3B-Instruct",
       api_base="http://localhost:8000/v1",
       key="not-needed",
       thread_num=512,
   )

How many concurrent API calls can I make?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This depends on your API provider:

- **OpenAI**: 50-200 (rate limits apply)
- **Local vLLM**: 256-1024 (depends on GPU memory)
- **SGLang cluster**: 2048-8192 (highest throughput)

Why are my MLLM calls slow?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common causes:

1. **Low thread_num**: Increase ``thread_num`` in model config
2. **API rate limiting**: Reduce ``thread_num`` or upgrade API tier
3. **Large images**: Use ``resize_image_size`` to reduce image size
4. **Network latency**: Consider local model deployment

Performance
-----------

How do I speed up processing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the :doc:`guide/quick_start` guide for performance tips. Key tips:

1. Pre-cache images with ``--cache-images``
2. Use rule-based filters before MLLM operators
3. Increase ``thread_num`` for MLLM operations
4. Use SSD storage for LMDB cache

What batch_size should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Available Memory
     - Recommended batch_size
   * - < 16 GB
     - 1,000 - 5,000
   * - 16-64 GB
     - 5,000 - 20,000
   * - > 64 GB
     - 20,000 - 100,000

How do I handle very large datasets (10M+ samples)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Pre-cache images** before running the pipeline
2. **Use config-driven pipelines** for automatic checkpointing
3. **Split processing** across multiple machines if needed
4. **Monitor progress** with W&B integration

Troubleshooting
---------------

My pipeline crashed. How do I resume?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply re-run the same command:

.. code-block:: bash

   python run.py -c my_config.py  # Automatically resumes

DataStudio saves checkpoints after each batch.

Why is my output empty?
~~~~~~~~~~~~~~~~~~~~~~~

Common causes:

1. **Filters too aggressive**: Check ``filter_ops`` in rejected samples
2. **Wrong data format**: Verify input matches expected format
3. **Path issues**: Check image paths are correct

How do I debug filtering decisions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examine the filtered samples:

.. code-block:: python

   kept, filtered = pipeline(data)

   for item in filtered[:10]:
       print(f"ID: {item['id']}")
       print(f"Filtered by: {item.get('filter_ops', {})}")
       print()

See :doc:`troubleshooting` for more detailed debugging guides.

Contributing
------------

How can I contribute?
~~~~~~~~~~~~~~~~~~~~~

We welcome contributions! See `CONTRIBUTING.md <https://github.com/Open-Bee/DataStudio/blob/main/CONTRIBUTING.md>`_ for guidelines:

- Bug reports and feature requests
- Documentation improvements
- New operators
- Performance optimizations

How do I report a bug?
~~~~~~~~~~~~~~~~~~~~~~

Open an issue on GitHub with:

1. DataStudio version
2. Python version and OS
3. Minimal reproducible example
4. Error message and traceback

Citation
--------

How should I cite DataStudio?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use DataStudio in your research, please cite:

.. code-block:: bibtex

   @article{zhang2025bee,
     title={Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs},
     author={Zhang, Yi and others},
     journal={arXiv preprint arXiv:2510.13795},
     year={2025}
   }

Still Have Questions?
---------------------

- Check the :doc:`troubleshooting` guide
- Browse the `GitHub Issues <https://github.com/Open-Bee/DataStudio/issues>`_
- Start a `Discussion <https://github.com/Open-Bee/DataStudio/discussions>`_
