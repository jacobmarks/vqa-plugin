"""VQA plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from PIL import Image
from transformers import pipeline


import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types


def get_filepath(sample):
    return (
        sample.local_path if hasattr(sample, "local_path") else sample.filepath
    )


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
        filepath = get_filepath(sample)
        image =  Image.open(filepath)

        question = ctx.params.get("question", "None provided")
        vqa_pipeline = pipeline("visual-question-answering")
        response = vqa_pipeline(image, question, top_k=1)

        return {
            "question": question,
            "answer": response[0]["answer"]
            }
       

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("question", label="Question")
        outputs.str("answer", label="Answer")
        header = "Visual Question Answering!"
        return types.Property(outputs, view=types.View(label=header))


def register(plugin):
    plugin.register(VQA)
