Troubleshooting Guide
=====================

This guide helps you diagnose and fix common issues with DataStudio.

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation Issues
-------------------

ImportError: No module named 'datastudio'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: python

   >>> from datastudio.operators import Pipeline
   ImportError: No module named 'datastudio'

**Solution:**

Install DataStudio in development mode:

.. code-block:: bash

   cd /path/to/DataStudio
   pip install -e .

Or ensure you're in the correct virtual environment:

.. code-block:: bash

   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   ModuleNotFoundError: No module named 'lmdb'

**Solution:**

.. code-block:: bash

   pip install -r requirements.txt

For specific packages:

.. code-block:: bash

   pip install lmdb pillow tqdm requests

Python Version Incompatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   SyntaxError: invalid syntax
   # or
   TypeError: 'type' object is not subscriptable

**Solution:**

DataStudio requires Python 3.10+. Check your version:

.. code-block:: bash

   python --version

Upgrade if needed:

.. code-block:: bash

   # Using pyenv
   pyenv install 3.10.12
   pyenv local 3.10.12

Data Loading Issues
-------------------

FileNotFoundError for Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   FileNotFoundError: [Errno 2] No such file or directory: 'images/sample.jpg'

**Causes & Solutions:**

1. **Relative paths**: Use absolute paths in your data or set ``data_root``

   .. code-block:: python

      # In dataset YAML
      data_root: /absolute/path/to/dataset

2. **Path format**: Ensure paths match your OS (forward vs backslashes)

3. **Missing files**: Verify images exist at specified paths

Invalid JSON Format
~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes

**Solution:**

Check your JSON/JSONL format:

.. code-block:: bash

   # Validate JSON
   python -m json.tool your_data.json

   # For JSONL, check each line
   python -c "
   import json
   with open('data.jsonl') as f:
       for i, line in enumerate(f, 1):
           try:
               json.loads(line)
           except json.JSONDecodeError as e:
               print(f'Error on line {i}: {e}')
   "

LMDB Cache Issues
~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   lmdb.Error: /path/to/cache: No such file or directory

**Solution:**

Ensure the parent directory exists:

.. code-block:: bash

   mkdir -p ~/cache/images_lmdb

**Symptom:**

.. code-block:: text

   lmdb.MapFullError: mdb_put: MDB_MAP_FULL: Environment mapsize limit reached

**Solution:**

Your LMDB database is full. Delete and recreate:

.. code-block:: bash

   rm -rf ~/cache/images_lmdb
   python run.py -c config.py --cache-images

Or use a different cache directory:

.. code-block:: python

   dataloader = dict(
       cache_dir="~/cache/images_lmdb_v2",  # New location
       # ...
   )

Pipeline Execution Issues
-------------------------

No Output / All Samples Filtered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:**

All samples are filtered out, output is empty.

**Diagnosis:**

Check why samples were filtered:

.. code-block:: python

   kept, filtered = pipeline(data)

   print(f"Kept: {len(kept)}, Filtered: {len(filtered)}")

   # Examine filter reasons
   from collections import Counter
   reasons = []
   for item in filtered:
       if 'filter_ops' in item:
           for op, qa_reasons in item['filter_ops'].items():
               for qa_idx, reason in qa_reasons.items():
                   reasons.append(f"{op}: {reason}")

   for reason, count in Counter(reasons).most_common(10):
       print(f"  {count:5d}x {reason}")

**Common causes:**

1. **Filters too strict**: Relax filter parameters
2. **Data format mismatch**: Check ``conversations`` structure
3. **Missing images**: Verify image paths

Pipeline Hangs
~~~~~~~~~~~~~~

**Symptom:**

Pipeline stops responding, no progress for extended time.

**Causes & Solutions:**

1. **MLLM API timeout**: Reduce ``thread_num`` or increase ``timeout``

   .. code-block:: python

      model = dict(
          thread_num=64,  # Reduce from 512
          timeout=(60, 3600),  # Increase timeout
      )

2. **Deadlock in threading**: Restart and reduce parallelism

3. **Memory exhaustion**: Reduce ``batch_size``

   .. code-block:: python

      dataloader = dict(
          batch_size=1000,  # Reduce from 10000
      )

Out of Memory (OOM)
~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   MemoryError
   # or
   Killed (process terminated by OS)

**Solutions:**

1. **Reduce batch size**:

   .. code-block:: python

      dataloader = dict(batch_size=1000)

2. **Resize images**:

   .. code-block:: python

      dataloader = dict(resize_image_size=1024)  # Smaller images

3. **Disable image loading** if not needed:

   .. code-block:: python

      dataloader = dict(use_image=False)

MLLM Issues
-----------

API Connection Errors
~~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8000)

**Solutions:**

1. **Check API server is running**:

   .. code-block:: bash

      curl http://localhost:8000/v1/models

2. **Verify API base URL** in config:

   .. code-block:: python

      model = dict(
          api_base="http://localhost:8000/v1",  # Include /v1
      )

3. **Check firewall/network settings**

Rate Limiting
~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   HTTP 429: Too Many Requests
   # or
   Rate limit exceeded

**Solution:**

Reduce concurrent requests:

.. code-block:: python

   model = dict(
       thread_num=50,  # Reduce from higher value
       retry=10,       # Add retries for transient errors
   )

Invalid API Response
~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   json.decoder.JSONDecodeError: Expecting value

**Diagnosis:**

Check the raw API response:

.. code-block:: python

   import requests
   response = requests.post(
       "http://localhost:8000/v1/chat/completions",
       json={"model": "test", "messages": [{"role": "user", "content": "hi"}]}
   )
   print(response.status_code)
   print(response.text)

**Common causes:**

1. Wrong API endpoint
2. Model not loaded on server
3. Invalid request format

Checkpoint & Resume Issues
--------------------------

Checkpoint Not Working
~~~~~~~~~~~~~~~~~~~~~~

**Symptom:**

Re-running doesn't resume from checkpoint.

**Diagnosis:**

Check if checkpoint file exists:

.. code-block:: bash

   ls -la work_dirs/my_experiment/checkpoint.json

**Solutions:**

1. **Ensure same work_dir**: Config must specify the same ``work_dir``

2. **Check checkpoint content**:

   .. code-block:: python

      import json
      with open("work_dirs/my_experiment/checkpoint.json") as f:
          print(json.dumps(json.load(f), indent=2))

3. **Manual checkpoint reset**: Delete checkpoint to restart:

   .. code-block:: bash

      rm work_dirs/my_experiment/checkpoint.json

Corrupted Checkpoint
~~~~~~~~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   json.decoder.JSONDecodeError when loading checkpoint

**Solution:**

Delete the corrupted checkpoint and restart:

.. code-block:: bash

   rm work_dirs/my_experiment/checkpoint.json
   python run.py -c config.py  # Restart from beginning

Output Issues
-------------

Output Format Changed
~~~~~~~~~~~~~~~~~~~~~

**Symptom:**

Output data has unexpected fields or structure.

**Explanation:**

DataStudio adds metadata fields:

- ``filter_ops``: Which filters applied and why
- ``rewrite_ops``: Which rewriters modified the content
- ``ori_answer``: Original answer before rewriting
- ``rejected``: Whether sample was filtered out

To get clean output, post-process:

.. code-block:: python

   def clean_output(item):
       """Remove DataStudio metadata from output."""
       clean = item.copy()
       for key in ['filter_ops', 'rewrite_ops', 'ori_answer', 'rejected', 'source_file']:
           clean.pop(key, None)
       return clean

Duplicate Samples in Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:**

Same sample appears multiple times in output.

**Possible causes:**

1. **Re-running without clearing output**: Output is appended
2. **Input has duplicates**: Check input data

**Solution:**

Clear output directory before re-running:

.. code-block:: bash

   rm -rf output/my_dataset/*
   python run.py -c config.py

Performance Issues
------------------

Slow Image Loading
~~~~~~~~~~~~~~~~~~

**Symptom:**

Pipeline is slow, mostly waiting on image loading.

**Solution:**

Pre-cache images to LMDB:

.. code-block:: bash

   python run.py -c config.py --cache-images

This only needs to be done once per dataset.

MLLM Throughput Low
~~~~~~~~~~~~~~~~~~~

**Symptom:**

MLLM operations are slower than expected.

**Diagnosis:**

Calculate actual throughput:

.. code-block:: python

   import time

   start = time.time()
   kept, filtered = pipeline(data[:100])
   elapsed = time.time() - start

   print(f"Throughput: {100/elapsed:.2f} samples/sec")

**Solutions:**

1. **Increase thread_num** (if API allows):

   .. code-block:: python

      model = dict(thread_num=1024)

2. **Use local model** instead of API

3. **Optimize prompts** (shorter prompts = faster)

Getting More Help
-----------------

If you can't resolve your issue:

1. **Check GitHub Issues**: `github.com/Open-Bee/DataStudio/issues <https://github.com/Open-Bee/DataStudio/issues>`_

2. **Enable debug logging**:

   .. code-block:: python

      import logging
      logging.basicConfig(level=logging.DEBUG)

3. **Collect diagnostic info**:

   .. code-block:: bash

      python -c "
      import sys
      import datastudio
      print(f'Python: {sys.version}')
      print(f'DataStudio: {datastudio.__version__}')
      "

4. **Open a new issue** with:

   - DataStudio version
   - Python version and OS
   - Full error traceback
   - Minimal config to reproduce
   - Sample data (if possible)

See Also
--------

- :doc:`faq` - Frequently asked questions
- :doc:`guide/quick_start` - Quick start guide
- `GitHub Discussions <https://github.com/Open-Bee/DataStudio/discussions>`_
