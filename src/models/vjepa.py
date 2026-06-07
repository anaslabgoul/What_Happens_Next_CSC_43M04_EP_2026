from __future__ import annotations

import re
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForVideoClassification, AutoVideoProcessor


# Labels locaux de ton dataset.
# Important : la classe 027 est absente.
# On garde les IDs locaux parce que ton dataset retourne ces labels-là.
DATASET_ID_TO_LABEL: Dict[int, str] = {
    0: "Closing something",
    1: "Covering something with something",
    2: "Dropping something into something",
    3: "Folding something",
    4: "Hitting something with something",
    5: "Holding something",
    6: "Moving something away from something",
    7: "Moving something closer to something",
    8: "Moving something down",
    9: "Moving something up",
    10: "Opening something",
    11: "Picking something up",
    12: "Pouring something into something",
    13: "Pouring something out of something",
    14: "Pretending to pick something up",
    15: "Pretending to pour something out of something but something is empty",
    16: "Pretending to put something into something",
    17: "Pretending to throw something",
    18: "Pulling something from left to right",
    19: "Pulling something from right to left",
    20: "Putting something behind something",
    21: "Putting something in front of something",
    22: "Putting something into something",
    23: "Putting something next to something",
    24: "Putting something onto something",
    25: "Showing something to the camera",
    26: "Spilling something next to something",
    28: "Taking something out of something",
    29: "Throwing something",
    30: "Turning something upside down",
    31: "Uncovering something",
    32: "Unfolding something",
}


def _normalize_label(text: str) -> str:
    """
    Normalise les labels pour matcher :
      - 'Closing [something]'
      - 'Closing something'
      - '000_Closing_something'

    vers une forme comparable.
    """
    text = str(text).strip().lower()

    # Retirer préfixe éventuel du type 000_
    text = re.sub(r"^\d+_", "", text)

    text = text.replace("_", " ")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace("-", " ")

    # Uniformiser espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _build_dataset_to_hf_logit_mapping(
    id2label: Dict[int, str],
    num_classes: int,
) -> torch.Tensor:
    """
    Construit un tenseur de taille num_classes.

    mapping[dataset_id] = hf_logit_id correspondant au label texte.

    Si une classe locale est absente, par exemple 027, mapping[27] = -1.
    """
    normalized_hf_label_to_id: Dict[str, int] = {}

    for raw_id, raw_label in id2label.items():
        hf_id = int(raw_id)
        normalized = _normalize_label(raw_label)
        normalized_hf_label_to_id[normalized] = hf_id

    mapping = torch.full(
        size=(num_classes,),
        fill_value=-1,
        dtype=torch.long,
    )

    missing = []

    for dataset_id, dataset_label in DATASET_ID_TO_LABEL.items():
        if dataset_id >= num_classes:
            raise ValueError(
                f"Dataset label id {dataset_id} >= num_classes={num_classes}. "
                "Garde num_classes à au moins 33."
            )

        normalized_dataset_label = _normalize_label(dataset_label)

        hf_id = normalized_hf_label_to_id.get(normalized_dataset_label)

        if hf_id is None:
            missing.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_label,
                    "normalized": normalized_dataset_label,
                }
            )
            continue

        mapping[dataset_id] = hf_id

    if missing:
        available_examples = list(normalized_hf_label_to_id.keys())[:20]
        raise ValueError(
            "Impossible de mapper certaines classes locales vers les labels "
            "du checkpoint V-JEPA SSV2.\n"
            f"Missing mappings: {missing}\n\n"
            f"Exemples de labels HF normalisés disponibles: {available_examples}\n\n"
            "Il faut probablement corriger DATASET_ID_TO_LABEL pour matcher "
            "exactement les labels id2label du checkpoint."
        )

    print("\nV-JEPA SSV2 dataset-id -> HF-logit-id mapping:")
    for dataset_id in range(num_classes):
        hf_id = int(mapping[dataset_id].item())
        if hf_id < 0:
            print(f"  local {dataset_id:03d}: INVALID / absent")
        else:
            label = DATASET_ID_TO_LABEL.get(dataset_id, "<unknown>")
            print(f"  local {dataset_id:03d} -> HF {hf_id:03d}: {label}")

    return mapping


class VJEPA2VideoClassifier(nn.Module):
    """
    V-JEPA2 ForVideoClassification pré-entraîné sur SSV2.

    Cette version utilise :
      - le backbone V-JEPA2 pré-entraîné ;
      - la tête de classification vidéo SSV2 pré-entraînée ;
      - un mapping texte pour aligner tes classes locales vers les logits HF.

    Input repo:
        video_batch: (B, T, C, H, W)

    Output:
        logits: (B, num_classes)

    Avec ton dataset :
        num_classes doit rester à 33, car les labels vont jusqu'à 32,
        même si la classe 027 est absente.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        hidden_dim: int = 512,       # gardé pour compatibilité YAML, non utilisé
        dropout: float = 0.0,        # gardé pour compatibilité YAML, non utilisé
        input_norm: str = "imagenet",
        target_num_frames: Optional[int] = 16,
        frame_resampling: str = "repeat",
    ) -> None:
        super().__init__()

        if not pretrained:
            raise ValueError(
                "Ce modèle doit utiliser un checkpoint V-JEPA2 SSV2 pré-entraîné. "
                "Mets model.pretrained: true."
            )

        if int(num_classes) < 33:
            raise ValueError(
                "Ton dataset a des labels jusqu'à 032 avec la classe 027 absente. "
                "Il faut garder num_classes: 33, pas 32."
            )

        self.num_classes = int(num_classes)
        self.model_name = str(model_name)
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_last_n_layers = int(unfreeze_last_n_layers)
        self.input_norm = str(input_norm)
        self.target_num_frames = target_num_frames
        self.frame_resampling = str(frame_resampling)

        self.processor = AutoVideoProcessor.from_pretrained(self.model_name)

        self.vjepa_model = AutoModelForVideoClassification.from_pretrained(
            self.model_name,
            attn_implementation="sdpa",
        )

        # Mapping robuste : labels locaux -> logits SSV2 HF.
        id2label = self.vjepa_model.config.id2label
        dataset_to_hf_logit = _build_dataset_to_hf_logit_mapping(
            id2label=id2label,
            num_classes=self.num_classes,
        )

        self.register_buffer(
            "dataset_to_hf_logit",
            dataset_to_hf_logit,
            persistent=False,
        )

        valid_dataset_ids = torch.tensor(
            [
                dataset_id
                for dataset_id in range(self.num_classes)
                if int(dataset_to_hf_logit[dataset_id].item()) >= 0
            ],
            dtype=torch.long,
        )

        valid_hf_ids = torch.tensor(
            [
                int(dataset_to_hf_logit[dataset_id].item())
                for dataset_id in range(self.num_classes)
                if int(dataset_to_hf_logit[dataset_id].item()) >= 0
            ],
            dtype=torch.long,
        )

        self.register_buffer(
            "valid_dataset_ids",
            valid_dataset_ids,
            persistent=False,
        )
        self.register_buffer(
            "valid_hf_ids",
            valid_hf_ids,
            persistent=False,
        )

        # Stats ImageNet utilisées par ton repo quand pretrained=True.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        # Stats du processor V-JEPA2.
        image_mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])

        self.register_buffer(
            "vjepa_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "vjepa_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        self._apply_freezing()

    def _find_transformer_layers(self) -> Optional[nn.ModuleList]:
        """
        Recherche robuste des blocs Transformer du modèle HF.
        """
        expected_layers = int(getattr(self.vjepa_model.config, "num_hidden_layers", 0))

        candidates = []
        for _, module in self.vjepa_model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                if expected_layers > 0 and len(module) == expected_layers:
                    candidates.append(module)

        if candidates:
            return candidates[-1]

        largest = None
        for _, module in self.vjepa_model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                if largest is None or len(module) > len(largest):
                    largest = module

        return largest

    def _apply_freezing(self) -> None:
        """
        freeze_backbone=True:
            - gèle tout le modèle V-JEPA2 ;
            - dégèle toujours le classifier SSV2 pré-entraîné ;
            - peut dégeler les N dernières couches Transformer.

        freeze_backbone=False:
            - fine-tune tout V-JEPA2 + classifier.
        """
        if not self.freeze_backbone:
            for parameter in self.vjepa_model.parameters():
                parameter.requires_grad = True
            return

        # Geler tout le modèle.
        for parameter in self.vjepa_model.parameters():
            parameter.requires_grad = False

        # Dégeler la tête de classification SSV2 pré-entraînée.
        if hasattr(self.vjepa_model, "classifier"):
            for parameter in self.vjepa_model.classifier.parameters():
                parameter.requires_grad = True
        else:
            print("Warning: vjepa_model.classifier introuvable.")

        # Dégeler éventuellement les dernières couches Transformer.
        if self.unfreeze_last_n_layers > 0:
            layers = self._find_transformer_layers()
            if layers is None:
                print(
                    "Warning: impossible de trouver les layers Transformer V-JEPA2. "
                    "Seul le classifier est entraîné."
                )
                return

            n = min(self.unfreeze_last_n_layers, len(layers))
            for layer in layers[-n:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

            print(f"V-JEPA2: unfroze last {n} transformer layers.")

    def _prepare_video(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch:
            (B, T, C, H, W)

        returns:
            pixel_values_videos: (B, T', C, H, W)
        """
        x = video_batch

        # Si ton repo a déjà normalisé en ImageNet, on revient en [0, 1].
        if self.input_norm == "imagenet":
            x = x * self.imagenet_std + self.imagenet_mean
            x = x.clamp(0.0, 1.0)
        elif self.input_norm == "none":
            x = x.clamp(0.0, 1.0)
        else:
            raise ValueError(
                f"input_norm={self.input_norm} non supporté. "
                "Utilise 'imagenet' ou 'none'."
            )

        if self.target_num_frames is not None:
            b, t, c, h, w = x.shape
            target_t = int(self.target_num_frames)

            if t != target_t:
                if self.frame_resampling == "repeat" and target_t % t == 0:
                    repeat_factor = target_t // t
                    x = x.repeat_interleave(repeat_factor, dim=1)

                elif self.frame_resampling == "nearest":
                    indices = torch.linspace(
                        0,
                        t - 1,
                        steps=target_t,
                        device=x.device,
                    ).round().long()
                    x = x[:, indices]

                elif self.frame_resampling == "interpolate":
                    x_perm = x.permute(0, 2, 3, 4, 1)  # (B, C, H, W, T)
                    x_perm = F.interpolate(
                        x_perm,
                        size=(h, w, target_t),
                        mode="trilinear",
                        align_corners=False,
                    )
                    x = x_perm.permute(0, 4, 1, 2, 3)  # (B, T, C, H, W)

                else:
                    raise ValueError(
                        f"frame_resampling={self.frame_resampling} non supporté. "
                        "Utilise 'repeat', 'nearest' ou 'interpolate'."
                    )

        x = (x - self.vjepa_mean) / self.vjepa_std
        return x

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        pixel_values_videos = self._prepare_video(video_batch)

        outputs = self.vjepa_model(pixel_values_videos=pixel_values_videos)
        full_logits = outputs.logits  # (B, nb_classes_ssv2)

        batch_size = full_logits.shape[0]

        # Sortie locale en 33 logits, avec 027 masqué automatiquement.
        logits = full_logits.new_full(
            size=(batch_size, self.num_classes),
            fill_value=-1e9,
        )

        logits[:, self.valid_dataset_ids.to(full_logits.device)] = full_logits.index_select(
            dim=1,
            index=self.valid_hf_ids.to(full_logits.device),
        )

        return logits