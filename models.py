from transformers import PaliGemmaForConditionalGeneration
import torch
import torch.nn as nn


class VQAClassifier(nn.Module):
    def __init__(self, base_model_id, num_labels, device=torch.device('cpu')):
        super(VQAClassifier, self).__init__()
        self.base_model = PaliGemmaForConditionalGeneration.from_pretrained(base_model_id, torch_dtype=torch.bfloat16).to(device)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        # Assuming the base model returns the last hidden state
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token representation
        logits = self.classifier(pooled_output)
        return logits

