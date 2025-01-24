import os
import hashlib

#############
# TPO Demo Utils # Adapted from LongVA Demo
#############

title_markdown = """
# [TPO Multimodal Chat](https://ruili33.github.io/tpo_website/)
"""

subtitle_markdown = """
### Web demo for TPO models from the paper Temporal Preference Optimization for Long-Form Video Understanding.
"""

html_header = """
<style>
/* Existing Styles for Larger Screens */
.header-container {
  display: flex;
  justify-content: left;
  align-items: center;
  text-align: left;
  background: linear-gradient(45deg, #00C0C0, #00C000);
  border-radius: 10px;
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.1);
  padding: 10px 20px; /* Added padding */
}

.header-container img {
  max-width: 80px;
  height: auto;
  border-radius: 10px;
}

.header-container a {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
}

/* Responsive adjustments for screens less than 768px wide */
@media (max-width: 768px) {
  .header-container {
    flex-direction: column;
    align-items: flex-start;
    padding: 10px 15px; /* Adjust padding for smaller screens */
  }

  .header-container img {
    max-width: 60px; /* Adjust image size for smaller screens */
  }

  .header-container h2, .header-container h5 {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
    text-align: center; /* Center text on small screens */
    margin-top: 5px; /* Add top margin for better spacing after stacking */
  }

  .header-container h2 {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
    font-size: 16px; /* Smaller font size for the title on mobile */
  }

  .header-container h5 {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
    font-size: 12px; /* Smaller font size for the subtitle on mobile */
  }

  .header-container a {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
  }
}
</style>

<div class="header-container">
 
  <div>
    <h2><a href="https://ruili33.github.io/tpo_website/">Temporal Preference Optimization for Long-Form Video Understanding</a></h2>
    <h5><a href="https://github.com/ruili33/TPO">Github</a> | <a href="https://huggingface.co/collections/ruili0/temporal-preference-optimization-67874b451f65db189fa35e10">Huggingface</a> | <a href="https://ruili33.github.io/tpo_website/">Project Page</a> | <a href="https://arxiv.org/abs/2501.13919v1">ArXiv Paper</a></h5>
  </div>
</div>
"""

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""





bibtext = """
## Citation
```
@misc{li2025temporalpreferenceoptimizationlongform,
      title={Temporal Preference Optimization for Long-Form Video Understanding}, 
      author={Rui Li and Xiaohan Wang and Yuhui Zhang and Zeyu Wang and Serena Yeung-Levy},
      year={2025},
      eprint={2501.13919},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.13919}, 
}
```
"""

PARENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
################## BACKEND ##################
os.environ["GRADIO_EXAMPLES_CACHE"] = (
    f"{PARENT_FOLDER}/cache"
)
os.environ["GRADIO_TEMP_DIR"] = (
    f"{PARENT_FOLDER}/cache"
)

def generate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:6]