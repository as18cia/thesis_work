import torch as torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("twmkn9/bert-base-uncased-squad2")

model = AutoModelForQuestionAnswering.from_pretrained("twmkn9/bert-base-uncased-squad2")

question, text = "private schools that charge no tuition called", "In the United Kingdom and several other Commonwealth countries including Australia and Canada, the use of the term is generally restricted to primary and secondary educational levels; it is almost never used of universities and other tertiary institutions. Private education in North America covers the whole gamut of educational activity, ranging from pre-school to tertiary level institutions. Annual tuition fees at K-12 schools range from nothing at so called 'tuition-free' schools to more than $45,000 at several New England preparatory schools."
inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))


# https://huggingface.co/models?sort=downloads&search=squad2
