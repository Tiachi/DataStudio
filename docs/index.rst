DataStudio Documentation
========================

.. image:: https://img.shields.io/badge/🐝-DataStudio-FFD700?style=for-the-badge&labelColor=000000
   :alt: DataStudio
   :align: center

**Industrial-Grade Multimodal Data Processing Pipeline for MLLMs**

DataStudio is the data curation engine behind the `Bee project <https://open-bee.github.io/>`_
and the `Honey-Data-15M <https://huggingface.co/datasets/Open-Bee/Honey-Data-15M>`_ dataset.

.. note::
   DataStudio is part of the Bee project, accepted to **ICLR 2026**.
   See the `paper <https://arxiv.org/abs/2510.13795>`_ for details.

Quick Links
-----------

.. list-table::
   :widths: 25 75

   * - 📄 `Paper <https://arxiv.org/abs/2510.13795>`_
     - Read the research paper on arXiv
   * - 🏠 `Project Page <https://open-bee.github.io/>`_
     - Visit the Bee project homepage
   * - 🤗 `Models & Data <https://huggingface.co/Open-Bee>`_
     - Access Bee-8B model and Honey-Data-15M dataset
   * - 💻 `GitHub <https://github.com/Open-Bee/DataStudio>`_
     - Source code and issue tracker

Key Features
------------

🧩 **Modular Architecture**
   16 built-in rule operators + MLLM operators with unified interface for easy composition

🤖 **MLLM-Powered Processing**
   Native integration with vision-language models for intelligent filtering and rewriting

⚡ **High Performance**
   Multi-process async concurrent API requests (8192+ via ``MPOpenAIAPI``) with LMDB image caching

📊 **Flexible Data Handling**
   JSON/JSONL support, multi-image samples, format preservation

Getting Started
---------------

Install DataStudio:

.. code-block:: bash

   git clone https://github.com/Open-Bee/DataStudio.git
   cd DataStudio
   pip install -r requirements.txt
   pip install -e .

Try the built-in examples:

.. code-block:: bash

   # Rule filtering example (no MLLM required)
   python run.py -c configs/examples/rule_filter_only.py

   # Text normalization example
   python run.py -c configs/examples/text_normalization.py

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   guide/quick_start
   guide/examples
   guide/architecture
   guide/development
   guide/multi_machine_deployment
   faq
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   architecture
   api/index

.. toctree::
   :maxdepth: 1
   :caption: 中文文档

   guide/readme_zh
   guide/quick_start_zh
   guide/examples_zh
   guide/architecture_zh

Community
---------

- **Bug Reports**: `GitHub Issues <https://github.com/Open-Bee/DataStudio/issues>`_
- **Feature Requests**: `GitHub Discussions <https://github.com/Open-Bee/DataStudio/discussions>`_
- **Contributing**: See `CONTRIBUTING.md <https://github.com/Open-Bee/DataStudio/blob/main/CONTRIBUTING.md>`_

Citation
--------

If you use DataStudio in your research, please cite:

.. code-block:: bibtex

   @article{zhang2025bee,
     title={Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs},
     author={Zhang, Yi and Ni, Bolin and Chen, Xin-Sheng and Zhang, Heng-Rui and Rao, Yongming and Peng, Houwen and Lu, Qinglin and Hu, Han and Guo, Meng-Hao and Hu, Shi-Min},
     journal={arXiv preprint arXiv:2510.13795},
     year={2025}
   }

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

----
