from model.Dynamic_network.DualPrompt import DualPrompt
from model.Regular.Tree_LoRA import Tree_LoRA
from model.Regular.HideLoRA import HideLoRA
from model.Regular.LwF import LwF
from model.Regular.EWC import EWC
from model.Regular.GEM import GEM
from model.Regular.OGD import OGD
from model.Replay.MbPAplusplus import MbPAplusplus
from model.Replay.LFPT5 import LFPT5
from model.Regular.O_LoRA import O_LoRA
from model.base_model import CL_Base_Model
from model.lora import lora

Method2Class = {"EWC"      : EWC,
                "GEM"      : GEM,  # Gradient Episodic Memory for Continual Learning
                "OGD"      : OGD,
                "LwF"      : LwF,  # Learning without Forgetting
                "DualPrompt": DualPrompt,
                "MbPA++"   : MbPAplusplus,  # Episodic Memory in Lifelong Language Learning
                "LFPT5"    : LFPT5,  # LFPT5: A Unified Framework for Lifelong Few-shot Language Learning Based on Prompt Tuning of T5
                "O_LoRA"   : O_LoRA,  # Orthogonal Subspace Learning for Language Model Continual Learning
                "Hide_LoRA" : HideLoRA,
                "Tree_LoRA": Tree_LoRA,
                "base"     : CL_Base_Model,
                "lora"     : lora}  # SeqLoRA

AllDatasetName = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

OLoRADatasetStandardName = ["dbpedia", "amazon", "yahoo", "agnews"]
