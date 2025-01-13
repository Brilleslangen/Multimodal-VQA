import numpy as np
import torch
from torch import LongTensor, FloatTensor, Tensor
from transformers import PaliGemmaPreTrainedModel, PaliGemmaForConditionalGeneration, PaliGemmaConfig, AutoModel
import torch.nn as nn
from transformers.models.paligemma.modeling_paligemma import PaliGemmaCausalLMOutputWithPast


class VQAClassifier(PaliGemmaForConditionalGeneration):
    def __init__(self, model_id, num_labels, **kwargs):
        config = PaliGemmaConfig.from_pretrained(model_id, **kwargs)
        super().__init__(config)

        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        print(config.hidden_size)
        self.classifier = nn.Linear(2304, num_labels)

    def forward(self, labels, **kwargs) -> tuple | dict:
        outputs = super().forward(output_hidden_states=True, **kwargs)
        print(outputs.hidden_states[-1])
        logits = self.classifier(outputs.hidden_states[-1])  # Shape: (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


import torch
import torch.nn as nn
from torch import LongTensor, FloatTensor, Tensor
from typing import Optional, Union, Dict

# If you're working in a local clone of HF Transformers or a custom environment,
# adjust these imports to point to the correct location of your PaliGemma code.
from transformers import (
    PaliGemmaConfig,
    PaliGemmaPreTrainedModel,
    AutoModel,
    AutoModelForCausalLM,
)
from transformers.utils import logging
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaMultiModalProjector,
)

logger = logging.get_logger(__name__)


class PaliGemmaForClassification(PaliGemmaPreTrainedModel):
    """
    A classification model built on top of PaliGemma. It:
      - Uses a vision tower + projector to obtain image embeddings.
      - Uses a language model (decoder-only Transformer) to encode text (and possibly inserted image embeddings).
      - Pools the final text hidden states (e.g., the first token) and applies a linear classifier.

    Args:
      model_id (str): path or ID to a pretrained PaliGemma checkpoint.
      num_labels (int): number of classification labels.
      **kwargs: passed to `PaliGemmaConfig.from_pretrained(...)`.
    """

    def __init__(self, model_id: str, num_labels: int, **kwargs):
        # 1) Load config
        config = PaliGemmaConfig.from_pretrained(model_id, **kwargs)
        super().__init__(config)

        # 2) Build submodules for vision + text (similar to PaliGemmaForConditionalGeneration)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

        # text_config is typically the "decoder" config
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        # We do NOT care about the LM head here; we just want the hidden states.

        # 3) Classification head
        # Some PaliGemma configs store `hidden_size` as a list, e.g. [vision_dim, text_dim].
        # We only need the text dimension for classification.
        self.classifier = nn.Linear(config.text_config.hidden_size, num_labels)

        # If your config includes a special token index for image insertion
        self.image_token_index = getattr(config, "image_token_index", None)
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1

        # Standard post-init hook
        self.post_init()

    # ------------------------------
    # Optional: hooking up embeddings
    # ------------------------------
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Classification model does not need the LM output embeddings
    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        pass

    # ------------------------------
    # Vision feature extraction
    # ------------------------------
    def get_image_features(self, pixel_values: FloatTensor) -> FloatTensor:
        """
        Forward pass through the vision tower, then project to the text dimension and scale.
        """
        # e.g., (batch_size, image_seq_len, vision_hidden_size)
        vision_outputs = self.vision_tower(pixel_values)
        vision_last_hidden = vision_outputs.last_hidden_state

        # Project to text_hidden_size
        projected = self.multi_modal_projector(vision_last_hidden)
        # Scale down by sqrt(text_hidden_size), matching original PaliGemma
        scaled = projected / np.sqrt(self.config.text_config.hidden_size**0.5)

        return scaled

    # ------------------------------
    # Forward for classification
    # ------------------------------
    def forward(
        self,
        input_ids: Optional[LongTensor] = None,
        pixel_values: Optional[FloatTensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        token_type_ids: Optional[LongTensor] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        labels: Optional[LongTensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = True,  # We want final hidden states
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-modal classification forward method:
          1) Merge image embeddings into text embeddings if pixel_values is provided.
          2) Forward pass through the language model.
          3) Pool the final hidden states (e.g. first token) -> classifier head.
          4) Return {"loss", "logits"}.

        Args:
          input_ids (LongTensor): shape (batch, seq_len) for text tokens.
          pixel_values (FloatTensor): optional images of shape (batch, channels, H, W).
          attention_mask (LongTensor): shape (batch, seq_len), 1 for tokens to attend to, 0 for masked/padded tokens.
          labels (LongTensor): shape (batch,); classification labels.
          token_type_ids, position_ids, inputs_embeds, etc.: optional.
          output_attentions (bool): if True, returns attention maps; not used for classification typically.
          output_hidden_states (bool): if True, returns intermediate hidden states from the LM.
          return_dict (bool): if True, returns a dict {"loss", "logits"}.

        Returns:
          A dict with:
            - "loss": cross-entropy classification loss if `labels` is provided, else None.
            - "logits": shape (batch_size, num_labels).
        """
        # Basic input checks
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Please provide either `input_ids` or `inputs_embeds`.")

        # If we are passing in pixel_values, we cannot also pass in custom inputs_embeds
        # (unless you do some advanced custom logic).
        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and inputs_embeds at the same time.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1) If the user did not provide direct embeddings, build them from input_ids
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2) If pixel_values are present, incorporate them at `image_token_index` positions
        if pixel_values is not None:
            if self.image_token_index is None:
                raise ValueError(
                    "No `image_token_index` found in the config, so we can't merge image embeddings. "
                    "Please ensure your config.image_token_index is set correctly."
                )

            # Extract image features from the vision tower
            image_features = self.get_image_features(pixel_values)
            # shape: (batch, image_seq_len, text_hidden_size)

            # We expect that in `input_ids`, each image corresponds to some number of image tokens.
            # Usually, you'd tokenize your text so that it has 1 or more special tokens for images,
            # e.g. <image>. Then, we find those positions in input_ids:
            special_image_mask = (input_ids == self.image_token_index).unsqueeze(-1)
            # shape: (batch, seq_len, 1), expanded to match (batch, seq_len, hidden_dim)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # 4) Forward pass through the text LM (decoder). We only need hidden_states for classification.
        lm_outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,  # force True so we can easily read .hidden_states
            **kwargs,
        )

        # last_hidden_state: (batch, seq_len, hidden_dim)
        last_hidden_state = lm_outputs.hidden_states[-1]

        # 5) Pooling strategy for classification:
        #    For a decoder-only model, there's no official "[CLS]" token.
        #    We'll just take the first token's embedding.
        #    (You could also do average pooling or the last token, depending on your usage.)
        cls_representation = last_hidden_state[:, 0, :]  # shape: (batch, hidden_dim)

        # 6) Classification
        logits = self.classifier(cls_representation)  # shape: (batch, num_labels)

        # 7) Optional classification loss
        loss = None
        if labels is not None:
            # E.g., shape of `labels`: (batch,)
            # shape of `logits`: (batch, num_labels)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        output = {"loss": loss, "logits": logits}

        if return_dict:
            return output
        else:
            # Return a tuple if not using dict
            return (loss, logits)

