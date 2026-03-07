Getting Started
===============

This guide will help you install DataStudio and run your first data processing pipeline.

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
-------------

Before installing DataStudio, ensure you have:

- **Python 3.10 or higher**
- **pip** package manager
- **Git** for cloning the repository
- Sufficient disk space for image caching (varies by dataset)

Installation
------------

Step 1: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/Open-Bee/DataStudio.git
   cd DataStudio

Step 2: Create Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows

Step 3: Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

Step 4: Install DataStudio
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e .

Step 5: Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import datastudio; print('✅ DataStudio installed successfully!')"

Optional: Weights & Biases
~~~~~~~~~~~~~~~~~~~~~~~~~~

For experiment tracking:

.. code-block:: bash

   pip install wandb
   wandb login

Quick Start
-----------

Option 1: Try Built-in Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After installation, run the built-in examples to verify your setup (no additional data needed):

.. code-block:: bash

   # Rule filtering example (CPU only, no MLLM required)
   python run.py -c configs/examples/rule_filter_only.py

   # Text normalization example (remove think tags, etc.)
   python run.py -c configs/examples/text_normalization.py

See the **Examples Guide** for all 5 examples covering different scenarios.

Option 2: Config-Driven (Production)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For production use, create a config file:

.. code-block:: python

   # my_pipeline.py
   _base_ = ["@/_base_/models/local_api_model.py", "@/_base_/dataset.py"]

   work_dir = "./work_dirs/my_experiment"

   logger = dict(type="Logger", log_file="logs/process.log")

   dataset_yaml = "/path/to/dataset.yaml"

   dataloader = dict(
       dataset=dataset_yaml,
       batch_size=1000,
       use_image=True,
       cache_dir="~/cache/images_lmdb",
   )

   datasaver = dict(
       dataset=dataset_yaml,
       output_dir="./output",
       save_yaml_name="processed",
   )

   pipeline = dict(
       type="Pipeline",
       operations={
           "filter": {
               "type": "FILTERS",
               "priority": 1,
               "cfg": {"type": "ConvLengthFilter", "min_length": 1, "max_length": 20}
           },
       }
   )

Then run:

.. code-block:: bash

   python run.py -c my_pipeline.py

Core Concepts
-------------

Data Format
~~~~~~~~~~~

DataStudio uses a standard multimodal conversation format:

.. code-block:: json

   {
       "id": "unique_id",
       "image": "path/to/image.jpg",
       "conversations": [
           {"from": "human", "value": "Question text"},
           {"from": "gpt", "value": "Answer text"}
       ]
   }

- ``id``: Unique sample identifier
- ``image``: Path to image file (string or list for multi-image)
- ``conversations``: List of messages, alternating between ``human`` and ``gpt``

Operators
~~~~~~~~~

DataStudio provides two types of operators:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Type
     - Purpose
     - Returns
   * - **Filter**
     - Decide to keep or remove samples
     - ``(rejected: bool, reason: str)``
   * - **Rewriter**
     - Modify content without removing
     - ``new_answer`` or ``None``

Pipeline
~~~~~~~~

A Pipeline combines multiple operators:

.. code-block:: python

   pipeline = Pipeline([
       filter1,    # Executes first
       filter2,    # Executes second
       rewriter1,  # Executes third
   ])

Samples filtered out by early operators skip later operators.

Built-in Operators
------------------

Filters
~~~~~~~

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
   * - ``ConvLengthFilter``
     - Filter by conversation turns
   * - ``ImageSizeFilter``
     - Filter by image dimensions
   * - ``ImageAspectRatioFilter``
     - Filter by aspect ratio
   * - ``TextRepeatFilter``
     - Detect text repetition
   * - ``MLLMFilter``
     - MLLM-powered filtering

Rewriters
~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
   * - ``RemoveThinkRewriter``
     - Remove ``<think>`` tags
   * - ``NormPromptRewriter``
     - Normalize prompts
   * - ``SplitRewriter``
     - Split multi-turn to single
   * - ``MLLMRewriter``
     - MLLM-powered rewriting

Dataset Configuration
---------------------

Create a YAML file to define your dataset:

.. code-block:: yaml

   # dataset.yaml
   data_root: /path/to/data

   datasets:
     - file_path: train.jsonl
       source: my_dataset
       split: train

     - file_path: eval.jsonl
       source: my_dataset
       split: eval

What's Next?
------------

Now that you have DataStudio running, explore:

- :doc:`guide/quick_start` - Complete quick start guide
- :doc:`guide/examples` - Ready-to-run example configurations
- :doc:`guide/development` - Creating custom operators
- :doc:`guide/architecture` - Deep dive into DataStudio internals

Need Help?
----------

- :doc:`faq` - Frequently asked questions
- :doc:`troubleshooting` - Common issues and solutions
- `GitHub Issues <https://github.com/Open-Bee/DataStudio/issues>`_ - Report bugs
