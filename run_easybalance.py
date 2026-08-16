####################################
# Environment Variables to set:
    # MODEL_ROOT_DIR: {MODEL_ROOT_DIR}/Qwen3-30B-A3B
    # DS_DIR: {DS_DIR}/LongBench

import os
os.environ["MODEL_ROOT_DIR"] = "/models"
os.environ["DS_DIR"] = "/data/LongBench"

####################################
os.environ["RAYON_NUM_THREADS"] = "128"
os.environ["TOKENIZERS_PARALLELISM"] = "true"    
import torch
import json
import time
import math
import datetime
from itertools import islice
from transformers import AutoTokenizer

from easyinfra.models.distributed import DistributedInferenceEngine
from easyinfra.utils import print_and_record
from easyinfra.generation.parallel.parallel_configuration import MoeCommMode


def get_model_path(model_name):
    model_root_dir = os.environ.get("MODEL_ROOT_DIR", "./models")
    return os.path.join(model_root_dir, model_name)
    
def batched(gen, batch_size):
    i = 0
    while True:
        if isinstance(gen, list):
            batch = gen[i:i+batch_size]
        else:
            raise NotImplementedError
            batch = list(islice(gen, batch_size))
        if not batch:
            break
        yield batch
        i += batch_size

def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    local_world_size = len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))
    world_size = 8
    model_dtype = torch.bfloat16
    
    SHOW_OUTPUT = True
        
    os.environ["ENABLE_COMPUTE_COMM_OVERLAP"] = "y"
    os.environ["ENABLE_MOE_SCHEDULER"] = "1"
    # os.environ["MOE_IMPL"] = "fused"
    # os.environ["MOE_IMPL"] = "native"
    os.environ["MOE_IMPL"] = "cutlass"
    os.environ["MOE_COMM_MODE"] = MoeCommMode.ALL2ALL.value    
        
    model_paths = [
        # get_model_path("Qwen3-30B-A3B-Instruct-2507"),
        # get_model_path("Qwen3-235B-A22B-Instruct-2507"),
        get_model_path("Moonlight-16B-A3B-Instruct"),
    ]
    
    DATASETS = ["2wikimqa", "gov_report", "hotpotqa", "lcc", "multifieldqa_en", \
                 "multi_news", "passage_count", "passage_retrieval_en", "qasper", \
                 "repobench-p", "samsum", "trec", "triviaqa", \
                 ] # 13 representative tasks
    DATASETS = ["2wikimqa"]
    # DATASETS = [("combined_" + dataset) for dataset in DATASETS]
    
    RECORD_PER_ITER = 40        
        
    ep_size = world_size
    attn_tp_size = 1
    shared_expert_tp_size = attn_tp_size
            
    cross_layer_scheduling_strategies = ("none","enumerate_max_util")
    # cross_layer_scheduling_strategies = ("none", "enumerate_max_util", "cumsum_max_util", "each_diff_peak",)
    NUM_MIN_RUN_CHUNK = (3,)
    NUM_MAX_DEFERRED_STEPS = (0,)
    #######################################
    USE_EPLB = False
    
    NUM_ITER = 1
    for model_dir in model_paths:
        model_name = os.path.basename(model_dir)
        
        if model_name == "Qwen3-235B-A22B-Instruct-2507":
            SEQ_LEN = 512
            batch_size = 12 * ep_size
            max_chunk_size = SEQ_LEN*3
        elif model_name in ("Qwen3-30B-A3B-Instruct-2507", "Moonlight-16B-A3B-Instruct"):
            SEQ_LEN = 4096
            batch_size = 16 * ep_size
            max_chunk_size = SEQ_LEN*4
        else:
            raise NotImplementedError
                    
        from_pretrained_kwargs = {
            "world_size": world_size,
            "use_tp": True,
            "torch_dtype": model_dtype,
        }
        from_pretrained_kwargs["local_world_size"] = local_world_size
        from_pretrained_kwargs["ep_size"] = ep_size
        from_pretrained_kwargs["attn_tp_size"] = attn_tp_size
        from_pretrained_kwargs["shared_expert_tp_size"] = shared_expert_tp_size
        from_pretrained_kwargs["use_eplb"] = USE_EPLB
        if USE_EPLB:
            if model_name in ("Qwen3-30B-A3B-Instruct-2507", "Qwen3-235B-A22B-Instruct-2507"):
                num_global_experts = 128 # Qwen3MoE-2507
            elif model_name == "Moonlight-16B-A3B-Instruct":
                num_global_experts = 64
            else:
                raise ValueError
            from_pretrained_kwargs["num_global_experts"] = num_global_experts
            os.environ["EPLB_CONFIG"] = f"{os.path.dirname(__file__)}/config/moe/{model_name}-{num_global_experts}-1-1-{world_size}-{EPLB_CONFIG_SOURCE}.jsonl"
                
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
        model = DistributedInferenceEngine.from_pretrained(
          model_dir, 
          **from_pretrained_kwargs
        )
                        
        for cross_layer_scheduling_strategy in cross_layer_scheduling_strategies:
            # run None for 1 num_min_run_chunks is enough
            num_min_run_chunks_list = NUM_MIN_RUN_CHUNK
            num_max_deferred_steps_list = NUM_MAX_DEFERRED_STEPS
            is_none_strategy = "none" in cross_layer_scheduling_strategy
            if is_none_strategy:
                num_min_run_chunks_list = (NUM_MIN_RUN_CHUNK[0],)
                num_max_deferred_steps_list = (0,)
            # run different num_min_run_chunks
            for num_min_run_chunks in num_min_run_chunks_list:
                for num_max_deferred_steps in num_max_deferred_steps_list:    
                    for dataset in DATASETS:
                                            
                        num_new_tokens = 1
                        do_sample = False
                        temperature = 0.8
                        enable_scheduler = True
                        offload_kv = True
                        max_input_len = SEQ_LEN
                        
                        test_start = 0
                        test_end = batch_size * NUM_ITER
                        # read data
                        data = []
                        ## Longbench Path
                        path = f"{os.environ.get("DS_DIR")}/data/{dataset}.jsonl"
                        with open(path, "r", encoding="utf-8") as f:
                            data_item_i = 0
                            for line in f:
                                if line.strip():
                                    if data_item_i >= test_start:
                                        data.append(json.loads(line))
                                    data_item_i += 1
                                    if data_item_i >= test_end:
                                        break
                        
                        ## Repeat data for enough
                        if data_item_i < (test_end - test_start):
                            print(f"too many items (start:{test_start}, end:{test_end}) for {dataset}, REPEAT!")
                            data *= (test_end - test_start) // data_item_i
                            if (test_end - test_start) % data_item_i != 0:
                                data += data[:(test_end - test_start) % data_item_i]
                        
                        prompts = [f"{data_item["context"]} {data_item["input"]}" for data_item in data]
                                                                    
                        fd = None
                        print_and_record(fd, f"model_dir: {model_dir}")
                        print_and_record(fd, f"dataset: {dataset}")
                        print_and_record(fd, f"test_start: {test_start}")
                        print_and_record(fd, f"test_end: {test_end}")
                        print_and_record(fd, f"num_new_tokens: {num_new_tokens}")
                        print_and_record(fd, f"do_sample: {do_sample}")
                        print_and_record(fd, f"temperature: {temperature}")
                        print_and_record(fd, f"world_size: {world_size}")
                        print_and_record(fd, f"cross_layer_scheduling_strategy: {cross_layer_scheduling_strategy}")
                        print_and_record(fd, f"num min chunk: {num_min_run_chunks}")
                        print_and_record(fd, f"max defer steps: {num_max_deferred_steps}")

                        torch.cuda.synchronize()
                        start = time.time()
                        
                        total_gen_length = 0
                        for i, this_batch_prompts in enumerate(batched(prompts, batch_size)):
                            
                            now_raw_time = time.time()
                            print_and_record(fd, f"test batch item: {i} time {now_raw_time - start}")
                            
                            inputs = tokenizer(
                                this_batch_prompts, 
                                add_special_tokens=False, 
                                max_length=max_input_len,
                                padding=True,
                                truncation=True,
                                padding_side='left',
                            )
                            inputs["input_ids"] = torch.tensor(inputs["input_ids"])
                            inputs["attention_mask"] = torch.ones(len(inputs["attention_mask"]), len(inputs["attention_mask"][0]))
                            
                            # write to shared mem
                            num_toks = inputs["input_ids"].shape[0] * inputs["input_ids"].shape[1]
                            if num_toks > model.shared_mem["input_ids"].shape[-1]:
                                raise ValueError(f"Too large workload {num_toks} for size {model.shared_mem["input_ids"].shape[-1]} memory.")
                            model.shared_mem["input_ids"][:num_toks] = inputs["input_ids"].view(-1)
                            model.shared_mem["attention_mask"][:num_toks] = inputs["attention_mask"].view(-1)
                            model.shared_mem["shape"][0] = inputs["input_ids"].shape[0]
                            model.shared_mem["shape"][1] = inputs["input_ids"].shape[1]
                            
                            input_length = inputs["input_ids"].shape[-1]
                            # max_chunk_size = input_length
                            
                                
                            res_dict = model.generate(
                                do_sample=do_sample,
                                temperature=temperature,
                                max_input_length=max_input_len,
                                max_new_tokens=num_new_tokens,
                                enable_scheduler=enable_scheduler,
                                cross_layer_scheduling_strategy=cross_layer_scheduling_strategy,
                                max_chunk_size=max_chunk_size,
                                num_min_run_chunks=num_min_run_chunks,
                                num_max_deferred_steps=num_max_deferred_steps,
                                offload_kv=offload_kv,
                                # pad_token_id = ?,
                            )
                            output_shape = tuple(model.shared_mem["shape"].tolist())
                            num_generated_tokens = output_shape[1]
                            output_ids = model.shared_mem["output_ids"][:(output_shape[0] * output_shape[1])].clone().view(output_shape)
                            total_run_time = res_dict["total_run_time"]
                            this_run_time = res_dict["this_run_time"]
                            
                            this_gen_length = num_generated_tokens
                            total_gen_length += this_gen_length
                            
                            if SHOW_OUTPUT:
                                MAX_SHOW_OUTPUT_NUM = 1
                                iter_num = min(output_ids.shape[0], MAX_SHOW_OUTPUT_NUM)
                                for batch_i in range(iter_num):
                                    output_text = tokenizer.decode(torch.cat((inputs["input_ids"][batch_i,-10:], output_ids[batch_i,:]), dim=-1), skip_special_tokens=False)
                                    print(f"batch {batch_i}: {output_text}")  
                                batch_i = -1                      
                                output_text = tokenizer.decode(torch.cat((inputs["input_ids"][batch_i,-10:], output_ids[batch_i,:]), dim=-1), skip_special_tokens=False)
                                print(f"batch {batch_i}: {output_text}")  
                                
                            # record statistics
                            # print_and_record(fd, f"generated length: {this_gen_length}, total length: {total_gen_length}")
                            # print_and_record(fd, f"this token/s ****: {this_gen_length / this_run_time}")
                                
                            if (i - test_start) % RECORD_PER_ITER == (RECORD_PER_ITER -1) and fd is not None:
                                fd.flush()
                                
                            if i >= test_end - 1:
                                break
                        
                            
                        torch.cuda.current_stream().synchronize()
                        period = time.time() - start
                        
                        print_and_record(fd, f"model.this_run_time: {this_run_time}, outside this time: {period}, model.total_run_time: {total_run_time}")
                        
                        if fd is not None:
                            fd.close()
                
        if isinstance(model, DistributedInferenceEngine):
            # if sum_generation_tokens != total_gen_length:
            #     print(f"sum_generation_tokens: {sum_generation_tokens}, total gen length: {total_gen_length}")
            worker_run_time, sum_generation_tokens, accept_num, candidate_num, assistant_model_run_time = model.get_generation_time()
            model.destroy()
        

if __name__ == "__main__":
    main()
