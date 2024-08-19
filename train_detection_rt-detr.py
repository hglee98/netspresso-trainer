# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from netspresso_trainer import train_cli


def train_with_inline_yaml(base_dir: str):
    from netspresso_trainer import train_with_yaml
    logging_dir = train_with_yaml(
        # gpus="0,1",
        data=f"{base_dir}/data.yaml",
        augmentation=f"{base_dir}/augmentation.yaml",
        model=f"{base_dir}/model.yaml",
        training=f"{base_dir}/training.yaml",
        logging=f"{base_dir}/logging.yaml",
        environment=f"{base_dir}/environment.yaml",
        log_level='INFO'
    )
    return logging_dir


if __name__ == '__main__':
    # logging_dir = train_cli()

    # With inline yaml
    base_dir = "config/benchmark_examples/detection-coco2017-rtdetr-res18"
    logging_dir = train_with_inline_yaml(base_dir)
    
    print(f"Training results are saved at: {logging_dir}")