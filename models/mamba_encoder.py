import torch
import torch.nn as nn
from transformers import MambaModel
from peft import get_peft_model, LoraConfig


class MambaEncoder(nn.Module):
    def __init__(self, input_dim, n_layers=4, dropout=0.1, use_lora=True, lora_rank=32):
        super().__init__()

        # 1. Load pretrained Mamba
        full_mamba = MambaModel.from_pretrained("state-spaces/mamba-130m-hf")
        self.config = full_mamba.config

        # 2. Truncate layers
        if n_layers < self.config.num_hidden_layers:
            print(f"Truncating Mamba from {self.config.num_hidden_layers} → {n_layers} layers")
            full_mamba.layers = full_mamba.layers[:n_layers]
            self.config.num_hidden_layers = n_layers

        self.mamba = full_mamba

        # -------------------------------
        # 3. Apply LoRA
        # -------------------------------
        if use_lora:
            print("Applying LoRA adapters to Mamba...")
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=16,
                target_modules=["in_proj", "x_proj", "dt_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=None
            )

            self.mamba = get_peft_model(self.mamba, peft_config)

            # Freeze base weights
            for name, param in self.mamba.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False

            trainable_params, all_params = self.mamba.get_nb_trainable_parameters()
            print(f"LoRA Trainable Params: {trainable_params:,} / {all_params:,}")

        # -------------------------------
        # 4. Project Input Channel → Hidden Dim
        # -------------------------------
        self.project_in = nn.Linear(input_dim, self.config.hidden_size)
        nn.init.xavier_uniform_(self.project_in.weight)
        nn.init.zeros_(self.project_in.bias)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, attention_mask=None):
        # x: [B, SeqLen, C]
        x = self.project_in(x)

        outputs = self.mamba(
            inputs_embeds=x,
            attention_mask=attention_mask
        )

        return self.dropout(outputs.last_hidden_state)

