import datasets
import os
import json


class H_Config(datasets.BuilderConfig):
    def __init__(self,
                 *args,
                 data_file=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_file = data_file

class HHH(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIG_CLASS = H_Config
    BUILDER_CONFIGS = [
        H_Config(name="3H", version=VERSION, description="Plain text"),
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "prompt":datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": self.config.data_file,
                    "split": "test",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir["dev"]),
            #         "split": "dev",
            #     },
            # ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        with open(filepath, encoding="utf-8") as dataset:
            dataset=json.load(dataset)
            idx = 0
            for sample in dataset:
                prompt = sample['prompt']
                        
                yield idx,{
                    "prompt":prompt,
                }
                idx+=1
        # else:
        #     with open(filepath, encoding="utf-8") as dataset:
        #         dataset=json.load(dataset)
        #         idx=0
        #         for subgroup in dataset:
        #             subgroup_name = subgroup['sub_group']
        #             prompts = subgroup['prompts']
        #             for sample in prompts:
        #                 prompt = sample['prompt']
        #                 yield idx,{
        #                 "prompt":prompt,
        #                 "sub_group":subgroup_name
        #                 }
        #                 idx+=1
                        
                    
