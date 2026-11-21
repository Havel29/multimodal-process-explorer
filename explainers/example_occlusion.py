"""Example explanation method: leave one sentence out, using nothing but model.predict.

Any model kind works with it. To use it:

    [explanation]
    model = "flava"
    method = "explainers/example_occlusion.py"
    label = "Sentence occlusion"
    options = { max_sentences = 40 }

See app/explainers.py for what explain() receives and returns.
"""
import re


def explain(model, history, image_path, target=None, with_image=True, max_sentences=40, **options):
    base = model.predict(history, image_path)
    if target is None:
        target = max(range(len(base)), key=base.__getitem__)

    tokens = []
    if model.spec.uses_text:
        # The most recent sentences only: the model keeps the end of a long history
        sentences = [match for match in re.finditer(r"[^.]+\.?", history) if match.group().strip()][-max_sentences:]
        for sentence in sentences:
            start, end = sentence.span()
            without = model.predict(history[:start] + history[end:], image_path)
            score = base[target] - without[target]
            # Spread the sentence's score over its words, so the explorer can colour them
            words = list(re.finditer(r"\S+", sentence.group()))
            tokens += [(start + word.start(), start + word.end(), score / len(words)) for word in words]

    patches = []
    if with_image and image_path is not None and model.spec.kind == "flava":
        # One cell: how much the image adds (the multimodal adapter blanks a missing image)
        patches = [[base[target] - model.predict(history, None)[target]]]
    return {"probabilities": base, "target": target, "tokens": tokens, "patches": patches}
