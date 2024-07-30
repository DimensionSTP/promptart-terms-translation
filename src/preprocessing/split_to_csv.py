import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import json

import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def split_to_csv(
    config: DictConfig,
) -> None:
    with open(
        f"{config.connected_dir}/data/dataset_{config.date}.json",
        "r",
        encoding="utf-8",
    ) as f:
        dataset = json.load(f)

    def create_instruction(row):
        domain = row["domain"]
        paper = row["paper"]
        terms = ", ".join(row["terms"])
        instruction = f"""
The following text is an excerpt from the field of {domain} in the paper "{paper}".
Translate the following content from English to Korean, ensuring that the terms "{terms}" are translated as specified, with the original English terms in parentheses.
"""
        return instruction

    if not os.path.exists(f"{config.connected_dir}/data/{config.date}"):
        os.makedirs(
            f"{config.connected_dir}/data/{config.date}",
            exist_ok=True,
        )

    for split in ["train", "valid", "test"]:
        df = pd.DataFrame(dataset[split])
        df["instruction"] = df.apply(
            create_instruction,
            axis=1,
        )
        df = df[
            [
                "instruction",
                "english",
                "korean",
            ]
        ]
        if split != "valid":
            df.to_csv(
                f"{config.connected_dir}/data/{config.date}/{split}.csv",
                index=False,
            )
        else:
            df.to_csv(
                f"{config.connected_dir}/data/{config.date}/val.csv",
                index=False,
            )


if __name__ == "__main__":
    split_to_csv()
