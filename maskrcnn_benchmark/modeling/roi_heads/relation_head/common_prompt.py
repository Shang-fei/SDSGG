import torch
from torch import nn


class CLIPTextCommonPromptEncoder(nn.Module):
    def __init__(self, clip_model, prompt_length, init_std=0.02):
        super(CLIPTextCommonPromptEncoder, self).__init__()
        self.clip_model = clip_model
        self.prompt_length = prompt_length
        self.context_length = clip_model.context_length
        self.width = clip_model.ln_final.weight.shape[0]

        self.common_prompt = nn.Parameter(torch.empty(prompt_length, self.width))
        nn.init.normal_(self.common_prompt, std=init_std)
        self._freeze_text_tower()

    def _freeze_text_tower(self):
        for module in (
            self.clip_model.token_embedding,
            self.clip_model.transformer,
            self.clip_model.ln_final,
        ):
            for param in module.parameters():
                param.requires_grad = False

        self.clip_model.positional_embedding.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

    def forward(self, text_tokens):
        if self.prompt_length <= 0:
            return self.clip_model.encode_text(text_tokens)

        if text_tokens.dim() != 2:
            raise ValueError("text_tokens must be a 2D tensor of shape [num_texts, context_length].")

        if text_tokens.size(1) != self.context_length:
            raise ValueError(
                "text_tokens context length {} does not match CLIP context length {}.".format(
                    text_tokens.size(1), self.context_length
                )
            )

        text_tokens = text_tokens.to(self.clip_model.positional_embedding.device)
        dtype = self.clip_model.dtype

        token_embeddings = self.clip_model.token_embedding(text_tokens).type(dtype)
        prompt = self.common_prompt.unsqueeze(0).expand(text_tokens.size(0), -1, -1).type(dtype)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : self.context_length - self.prompt_length, :]
        prompted = torch.cat((prefix, prompt, suffix), dim=1)

        prompted = prompted + self.clip_model.positional_embedding.type(dtype)
        prompted = prompted.permute(1, 0, 2)
        prompted = self.clip_model.transformer(prompted)
        prompted = prompted.permute(1, 0, 2)
        prompted = self.clip_model.ln_final(prompted).type(dtype)

        eot_indices = torch.clamp(text_tokens.argmax(dim=-1) + self.prompt_length, max=self.context_length - 1)
        prompted = prompted[torch.arange(prompted.shape[0], device=prompted.device), eot_indices]
        prompted = prompted @ self.clip_model.text_projection
        return prompted
