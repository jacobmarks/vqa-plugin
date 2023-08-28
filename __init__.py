"""VQA plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
import os
from PIL import Image

import fiftyone as fo
import fiftyone.core.utils as fou
import fiftyone.operators as foo
from fiftyone.operators import types


transformers = fou.lazy_import("transformers")
replicate = fou.lazy_import("replicate")


def allows_replicate_models():
    """Returns whether the current environment allows replicate models."""
    return (
        find_spec("replicate") is not None
        and "REPLICATE_API_TOKEN" in os.environ
    )


def allows_hf_models():
    """
    Returns whether the current environment allows hugging face transformer
    models.
    """
    return find_spec("transformers") is not None


def get_filepath(sample):
    return (
        sample.local_path if hasattr(sample, "local_path") else sample.filepath
    )


class VQAModel:
    """Wrapper around a VQA model."""

    def __init__(self):
        pass

    # def _get_image(self, sample):
    #     filepath = get_filepath(sample)
    #     image = Image.open(filepath)
    #     return image

    def __call__(self, sample, question):
        pass


class ViLTVQAModel(VQAModel):
    """Wrapper around a Hugging Face ViLT VQA model."""

    def __call__(self, sample, question):
        filepath = get_filepath(sample)
        image = Image.open(filepath)
        from transformers import pipeline

        vqa_pipeline = pipeline("visual-question-answering")
        response = vqa_pipeline(image, question, top_k=1)
        return response[0]["answer"]


class BLIP2VQAModel(VQAModel):
    """Wrapper around Replicate BLIP2 VQA model."""

    def __call__(self, sample, question):
        filepath = get_filepath(sample)
        response = replicate.run(
            "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
            input={"image": open(filepath, "rb"), "question": question},
        )
        return response


def _get_vqa_model(model_name):
    if model_name == "vilt":
        return ViLTVQAModel()
    if model_name == "blip2":
        return BLIP2VQAModel()

    raise ValueError(f"Model {model_name} not found.")


def run_vqa(sample, question, model_name):
    model = _get_vqa_model(model_name)
    return model(sample, question)


class VQA(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="answer_visual_question",
            label="VQA: Answer question about selected image",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="VQA", description="Ask a question about the selected image!"
        )

        blip_flag = allows_replicate_models()
        vilt_flag = allows_hf_models()
        if not blip_flag and not vilt_flag:
            inputs.message(
                "message",
                label="No models available. Please set up your environment variables.",
            )
            return types.Property(inputs)

        radio_choices = types.RadioGroup()
        if blip_flag:
            radio_choices.add_choice("blip2", label="BLIP2")
        if vilt_flag:
            radio_choices.add_choice("vilt", label="ViLT")

        inputs.enum(
            "model_name",
            radio_choices.values(),
            default=radio_choices.choices[0].value,
            label="Model",
            view=radio_choices,
        )

        num_selected = len(ctx.selected)
        if num_selected == 0:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(
                    label=f"You must select a sample to use this operator"
                ),
            )
        elif num_selected > 1:
            inputs.str(
                "too_many_samples_warning",
                view=types.Warning(
                    label=f"You must select only one sample to use this operator"
                ),
            )

        else:
            inputs.str("question", label="Question", required=True)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        sample = ctx.dataset[ctx.selected[0]]
        question = ctx.params.get("question", "None provided")
        model_name = ctx.params.get("model_name", "blip2")
        answer = run_vqa(sample, question, model_name)

        return {"question": question, "answer": answer}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("question", label="Question")
        outputs.str("answer", label="Answer")
        header = "Visual Question Answering!"
        return types.Property(outputs, view=types.View(label=header))


def register(plugin):
    plugin.register(VQA)
