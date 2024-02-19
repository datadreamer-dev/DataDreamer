:hide-toc:

DataDreamer
#######################################################

.. raw:: html

   <p align="center">
   <b>Prompt &nbsp;&nbsp;·&nbsp;&nbsp; Generate Synthetic Data <span id="tagline-last-separator">&nbsp;&nbsp;·&nbsp;&nbsp; </span><span style="white-space:nowrap">Train & Align Models</span></b><br /><br />
   <a href="https://github.com/datadreamer-dev/DataDreamer/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/datadreamer-dev/DataDreamer/release.yml?logo=githubactions&logoColor=white&label=Tests%20%26%20Release" alt="Tests & Release" style="max-width: 100%;"></a>
   <a href="https://codecov.io/gh/datadreamer-dev/DataDreamer"><img src="https://codecov.io/gh/datadreamer-dev/DataDreamer/graph/badge.svg?token=KZB00BKWJE"/></a>
   <a href="https://github.com/datadreamer-dev/DataDreamer/actions/workflows/tests.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/57b6a8cedd26481516a1a6af510d6b24272d0a76/assets/badge/v2.json" alt="Ruff" style="max-width: 100%;"></a>
   <a href="https://pypi.org/project/datadreamer.dev/"><img src="https://badge.fury.io/py/datadreamer.dev.svg"/></a>
   <a href="https://datadreamer.dev/docs/"><img src="https://img.shields.io/website.svg?down_color=red&down_message=offline&label=Documentation&up_message=online&url=https://datadreamer.dev/docs/"/></a>
   <a href="https://datadreamer.dev/docs/latest/pages/contributing.html"><img src="https://img.shields.io/badge/Contributor-Guide-blue?logo=Github&color=purple"/></a>
   <br />
   <a href="https://github.com/datadreamer-dev/DataDreamer/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
   <a href="https://ajayp.app/"><img src="https://img.shields.io/badge/NLP-NLP?labelColor=011F5b&color=990000&label=University%20of%20Pennsylvania"/></a>
   <a href="https://arxiv.org/abs/2402.10379"><img src="https://img.shields.io/badge/arXiv-2402.10379-b31b1b.svg"/></a>
   <a href="https://discord.gg/dwWW8wuCtK"><img src="https://img.shields.io/badge/Discord-Chat-blue?logo=discord&color=4338ca&labelColor=black"/></a>
   </p>

   <p>
      DataDreamer is a powerful open-source Python library for prompting, synthetic data generation, and training workflows. It is designed to be simple, extremely efficient, and research-grade.
   </p>

   <table class="docutils align-default demo-example">
   <tbody>
      <tr>
         <td colspan="2">
            <div id="installation">
               <b>Installation:</b>
               <div class="highlight-shell notranslate">
                  <div class="highlight">
                     <pre id="codecell0"><span></span>pip3 <span class="w"></span>install <span class="w"></span>datadreamer.dev</pre>
                  </div>
               </div>
            </div>
         </td>
      </tr>
   </tbody>
   <tbody>
      <tr>
         <th class="head"><pre>demo.py</pre></th>
         <th class="head">Result of <pre style="display: inline;">demo.py</pre></th>
      </tr>
   </tbody>
   <tbody>
      <tr>
         <td>
   <div class="literal-block-wrapper docutils container" id="id1">
   <div class="code-block-caption"><span class="caption-text">Train a model to generate a tweet summarizing a research paper abstract using synthetic data.</span><a class="headerlink" href="#id1" title="Permalink to this code">#</a></div>
   <div class="highlight-python notranslate"><div class="highlight"><pre><span></span><span class="kn">from</span> <span class="nn">datadreamer</span> <span class="kn">import</span> <span class="n">DataDreamer</span>
   <span class="kn">from</span> <span class="nn">datadreamer.llms</span> <span class="kn">import</span> <span class="n">OpenAI</span>
   <span class="kn">from</span> <span class="nn">datadreamer.steps</span> <span class="kn">import</span> <span class="n">DataFromPrompt</span><span class="p">,</span> <span class="n">ProcessWithPrompt</span>
   <span class="kn">from</span> <span class="nn">datadreamer.trainers</span> <span class="kn">import</span> <span class="n">TrainHFFineTune</span>
   <span class="kn">from</span> <span class="nn">peft</span> <span class="kn">import</span> <span class="n">LoraConfig</span>

   <span class="k">with</span> <span class="n">DataDreamer</span><span class="p">(</span><span class="s2">&quot;./output&quot;</span><span class="p">):</span>
      <span class="c1"># Load GPT-4</span>
      <span class="n">gpt_4</span> <span class="o">=</span> <span class="n">OpenAI</span><span class="p">(</span><span class="n">model_name</span><span class="o">=</span><span class="s2">&quot;gpt-4&quot;</span><span class="p">)</span>

      <span class="c1"># Generate synthetic arXiv-style research paper abstracts with GPT-4</span>
      <span class="n">arxiv_dataset</span> <span class="o">=</span> <span class="n">DataFromPrompt</span><span class="p">(</span>
         <span class="s2">&quot;Generate Research Paper Abstracts&quot;</span><span class="p">,</span>
         <span class="n">args</span><span class="o">=</span><span class="p">{</span>
            <span class="s2">&quot;llm&quot;</span><span class="p">:</span> <span class="n">gpt_4</span><span class="p">,</span>
            <span class="s2">&quot;n&quot;</span><span class="p">:</span> <span class="mi">1000</span><span class="p">,</span>
            <span class="s2">&quot;temperature&quot;</span><span class="p">:</span> <span class="mf">1.2</span><span class="p">,</span>
            <span class="s2">&quot;instruction&quot;</span><span class="p">:</span> <span class="p">(</span>
               <span class="s2">&quot;Generate an arXiv abstract of an NLP research paper.&quot;</span>
               <span class="s2">&quot; Return just the abstract, no titles.&quot;</span>
            <span class="p">),</span>
         <span class="p">},</span>
         <span class="n">outputs</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;generations&quot;</span><span class="p">:</span> <span class="s2">&quot;abstracts&quot;</span><span class="p">},</span>
      <span class="p">)</span>

      <span class="c1"># Ask GPT-4 to convert the abstracts to tweets</span>
      <span class="n">abstracts_and_tweets</span> <span class="o">=</span> <span class="n">ProcessWithPrompt</span><span class="p">(</span>
         <span class="s2">&quot;Generate Tweets from Abstracts&quot;</span><span class="p">,</span>
         <span class="n">inputs</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;inputs&quot;</span><span class="p">:</span> <span class="n">arxiv_dataset</span><span class="o">.</span><span class="n">output</span><span class="p">[</span><span class="s2">&quot;abstracts&quot;</span><span class="p">]},</span>
         <span class="n">args</span><span class="o">=</span><span class="p">{</span>
            <span class="s2">&quot;llm&quot;</span><span class="p">:</span> <span class="n">gpt_4</span><span class="p">,</span>
            <span class="s2">&quot;instruction&quot;</span><span class="p">:</span> <span class="p">(</span>
               <span class="s2">&quot;Given the abstract, write a tweet to summarize the work.&quot;</span>
            <span class="p">),</span>
            <span class="s2">&quot;top_p&quot;</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">,</span>
         <span class="p">},</span>
         <span class="n">outputs</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;inputs&quot;</span><span class="p">:</span> <span class="s2">&quot;abstracts&quot;</span><span class="p">,</span> <span class="s2">&quot;generations&quot;</span><span class="p">:</span> <span class="s2">&quot;tweets&quot;</span><span class="p">},</span>
      <span class="p">)</span>

      <span class="c1"># Create training data splits</span>
      <span class="n">splits</span> <span class="o">=</span> <span class="n">abstracts_and_tweets</span><span class="o">.</span><span class="n">splits</span><span class="p">(</span><span class="n">train_size</span><span class="o">=</span><span class="mf">0.90</span><span class="p">,</span> <span class="n">validation_size</span><span class="o">=</span><span class="mf">0.10</span><span class="p">)</span>

      <span class="c1"># Train a model to convert research paper abstracts to tweets</span>
      <span class="c1"># with the synthetic dataset</span>
      <span class="n">trainer</span> <span class="o">=</span> <span class="n">TrainHFFineTune</span><span class="p">(</span>
         <span class="s2">&quot;Train an Abstract =&gt; Tweet Model&quot;</span><span class="p">,</span>
         <span class="n">model_name</span><span class="o">=</span><span class="s2">&quot;google/t5-v1_1-base&quot;</span><span class="p">,</span>
         <span class="n">peft_config</span><span class="o">=</span><span class="n">LoraConfig</span><span class="p">(),</span>
      <span class="p">)</span>
      <span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">(</span>
         <span class="n">train_input</span><span class="o">=</span><span class="n">splits</span><span class="p">[</span><span class="s2">&quot;train&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">output</span><span class="p">[</span><span class="s2">&quot;abstracts&quot;</span><span class="p">],</span>
         <span class="n">train_output</span><span class="o">=</span><span class="n">splits</span><span class="p">[</span><span class="s2">&quot;train&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">output</span><span class="p">[</span><span class="s2">&quot;tweets&quot;</span><span class="p">],</span>
         <span class="n">validation_input</span><span class="o">=</span><span class="n">splits</span><span class="p">[</span><span class="s2">&quot;validation&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">output</span><span class="p">[</span><span class="s2">&quot;abstracts&quot;</span><span class="p">],</span>
         <span class="n">validation_output</span><span class="o">=</span><span class="n">splits</span><span class="p">[</span><span class="s2">&quot;validation&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">output</span><span class="p">[</span><span class="s2">&quot;tweets&quot;</span><span class="p">],</span>
         <span class="n">epochs</span><span class="o">=</span><span class="mi">30</span><span class="p">,</span>
         <span class="n">batch_size</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
      <span class="p">)</span>

      <span class="c1"># Publish and share the synthetic dataset</span>
      <span class="n">abstracts_and_tweets</span><span class="o">.</span><span class="n">publish_to_hf_hub</span><span class="p">(</span>
         <span class="s2">&quot;datadreamer-dev/abstracts_and_tweets&quot;</span><span class="p">,</span>
         <span class="n">train_size</span><span class="o">=</span><span class="mf">0.90</span><span class="p">,</span>
         <span class="n">validation_size</span><span class="o">=</span><span class="mf">0.10</span><span class="p">,</span>
      <span class="p">)</span>

      <span class="c1"># Publish and share the trained model</span>
      <span class="n">trainer</span><span class="o">.</span><span class="n">publish_to_hf_hub</span><span class="p">(</span><span class="s2">&quot;datadreamer-dev/abstracts_to_tweet_model&quot;</span><span class="p">)</span>
   </pre></div>
   </div>
   </div>
         </td>
         <td>
            <p align="center" class="animation-wrapper">
               <a href="https://datadreamer.dev/docs/latest/_static/images/demo.svg"><img src="https://datadreamer.dev/docs/latest/_static/images/demo.svg" alt="Demo" /></a>
               <br/>
               <br/>
               See the <a class="reference external" href="https://huggingface.co/datasets/datadreamer-dev/abstracts_and_tweets">synthetic dataset</a> and <a class="reference external" href="https://huggingface.co/datadreamer-dev/abstracts_to_tweet_model">the trained model</a>
            </p>
         </td>
      </tr> 
   </tbody>
   <tbody>
      <tr>
         <td colspan="2">
            <p align="center">
               🚀 For more demonstrations and recipes see the <a class="reference external" href="https://datadreamer.dev/docs/latest/pages/get_started/quick_tour/index.html" title="Quick Tour"> Quick Tour</a> page.
            </p>
         </td>
      </tr>
   </tbody>
   </table>
   <br />

With DataDreamer you can:

* 💬 **Create Prompting Workflows**: Create and run multi-step, complex, prompting workflows easily with major open source or API-based LLMs.
* 📊 **Generate Synthetic Datasets**: Generate synthetic datasets for novel tasks or augment existing datasets with LLMs.
* ⚙️ **Train Models**: Align models. Fine-tune models. Instruction-tune models. Distill models. Train on existing data or synthetic data.
* ... learn more about what's possible in the :doc:`Overview Guide <./pages/get_started/overview_guide>`.

DataDreamer is:

* 🧩 **Simple**: Simple and approachable to use with sensible defaults, yet powerful with support for bleeding edge techniques.
* 🔬 **Research-Grade**: Built for researchers, by researchers, but accessible to all. A focus on correctness, best practices, and reproducibility.
* 🏎️ **Efficient**: Aggressive caching and resumability built-in. Support for techniques like quantization, parameter-efficient training (LoRA), and more.
* 🔄 **Reproducible**: Workflows built with DataDreamer are easily shareable, reproducible, and extendable.
* 🤝 **Makes Sharing Easy**: Publishing datasets and models is simple. Automatically generate data cards and model cards with metadata. Generate a list of any citations required.
* ... learn more about the :doc:`motivation and design principles behind DataDreamer <./pages/get_started/motivation_and_design>`.

Citation
========

Please cite the `DataDreamer paper <https://arxiv.org/abs/2402.10379>`_:

.. code-block:: bibtex

   @misc{patel2024datadreamer,
      title={DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows}, 
      author={Ajay Patel and Colin Raffel and Chris Callison-Burch},
      year={2024},
      eprint={2402.10379},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
   }

Contact
=======

Please reach out to us via `email (ajayp@upenn.edu) <mailto:ajayp@upenn.edu>`_ or on `Discord <https://discord.gg/dwWW8wuCtK>`_ if you have any questions, comments, or feedback.


.. toctree::
   :hidden:
   :caption: Get Started

   🎉 Installation <pages/get_started/installation>
   🚀 Quick Tour <pages/get_started/quick_tour/index>
   💡 Motivation and Design <pages/get_started/motivation_and_design>
   📖 Overview Guide <pages/get_started/overview_guide>
   🎓 Advanced Usage <pages/advanced_usage/index>

.. toctree::
   :hidden:
   :caption: References

   API Reference <datadreamer>
   Index <modindex>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: About

   GitHub <https://github.com/datadreamer-dev/DataDreamer>
   PyPI <https://pypi.org/project/datadreamer.dev/>
   License <https://github.com/datadreamer-dev/DataDreamer/blob/main/LICENSE.txt>
   Citation <https://datadreamer.dev/docs/latest/#citation>
   Contact <https://datadreamer.dev/docs/latest/#contact>
   Contributing <pages/contributing>
