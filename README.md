## Visual Question Answering Plugin

![vqa_updated](https://github.com/jacobmarks/vqa-plugin/assets/12500356/67819454-19e3-4b4a-861f-afed465f4866)

### Updates

- **2024-05-07**: Major updates:

  - Added support for Moondream2 model.
  - Added support for reading question from field on the sample.
  - Added support for storing the answer in a field on the sample.
  - Added support for applying to all samples in the current view (one at a time).
  - Added support for delegated execution.
  - Added support for Python operator execution.

- **2024-05-03**: [@harpreetsahota204](https://github.com/harpreetsahota204) added support for Idefics-8b model from
  [Replicate](https://replicate.com/).
- **2023-10-24**: Added support for Llava-13b and Fuyu-8b models from
  [Replicate](https://replicate.com/).

## Plugin Overview

This plugin is a Python plugin that allows you to answer visual questions about
images in your dataset!

### Supported Models

This version of the plugin supports the following models:

- [Fuyu-8b](https://replicate.com/lucataco/fuyu-8b/) from Adept AI (via [Replicate](https://replicate.com/))
- [Llava-13b](https://replicate.com/yorickvp/llava-13b) (via [Replicate](https://replicate.com/))
- [ViLT](https://huggingface.co/transformers/model_doc/vilt.html) (default Vision Language Transformer used in the [Visual Question Answering pipeline](https://huggingface.co/tasks/visual-question-answering))
- [BLIPv2](https://replicate.com/andreasjansson/blip-2) (via [Replicate](https://replicate.com/))
- [Idefics2-8b](https://replicate.com/lucataco/idefics-8b) from Hugging Face (via [Replicate](https://replicate.com/))
- Moondream2 via [Hugging Face Transformers](https://huggingface.co/vikhyatk/moondream2) and via [Replicate](https://replicate.com/lucataco/moondream2)

Feel free to fork this plugin and add support for other models!

## Watch On Youtube

[![Video Thumbnail](https://img.youtube.com/vi/agNvjKH9rIQ/0.jpg)](https://www.youtube.com/watch?v=agNvjKH9rIQ&list=PLuREAXoPgT0RZrUaT0UpX_HzwKkoB-S9j&index=3)

## Installation

### Pre-requisites

1. If you plan to use it, install the Hugging Face transformers library:

```shell
pip install transformers
```

2. If you plan to use it, install the Replicate library:

```shell
pip install replicate
```

And add your Replicate API key to your environment:

```shell
export REPLICATE_API_TOKEN=<your-api-token>
```

### Install the plugin

```shell
fiftyone plugins download https://github.com/jacobmarks/vqa-plugin
```

## Operators

### `answer_visual_question`

- Applies the selected visual question answering model to the selected sample in
  your dataset and outputs the answer.

## Usage

The recommended interactive way to use this plugin is in the FiftyOne App with exactly one sample selected.

### Python Operator Execution

If you want to loop over samples in your dataset or view, you may be interested in using the Python operator execution mode.

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart", max_samples=5)

## Access the operator via its URI (plugin name + operator name)
vqa = foo.get_operator("@jacobmarks/vqa/answer_visual_question")

## Apply the operator to the dataset
vqa(
    dataset,
    model_name="llava",
    question="Describe the image",
    answer_field="llava_answer",
)

## Print the answers
print(dataset.values("llava_answer"))
```
