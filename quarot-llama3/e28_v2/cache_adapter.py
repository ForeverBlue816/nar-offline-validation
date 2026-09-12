"""Common metadata optimization for equal-length, unmasked batches only.

No attention/quantization arithmetic changes. Metadata is computed once from
known Python length per step rather than reconstructed twice per layer with
GPU truth-value/scalar extraction. The legacy wrapper is available separately.
"""
import torch
from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit

class SharedMetadataCache(MultiLayerPagedKVCache4Bit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._metadata_length=None; self._spec={}
        self.graph_mode=False
    def get_cache_specs_for_flash_infer(self,attention_mask):
        if attention_mask is not None: return super().get_cache_specs_for_flash_infer(attention_mask)
        if self.graph_mode: return self._spec
        if self._metadata_length!=self.length:
            count=self.page_cnt_from_length(self.length)
            self._spec={'kv_data':self.pages,'kv_param':self.scales,
                'kv_indptr':torch.arange(self.batch_size+1,device=self.device,dtype=torch.int32)*count,
                'kv_indices':(torch.arange(count,device=self.device,dtype=torch.int32)[None,:]*self.batch_size+torch.arange(self.batch_size,device=self.device,dtype=torch.int32)[:,None]).reshape(-1),
                'last_page_offset':torch.full((self.batch_size,),(((self.length-1)%self.page_size)+1) if self.length else 0,device=self.device,dtype=torch.int32)}
            self._metadata_length=self.length
        return self._spec

def make_cache(model,batch,capacity,legacy=False,page_size=None):
    cls=MultiLayerPagedKVCache4Bit if legacy else SharedMetadataCache
    return cls(batch_size=batch,page_size=page_size or capacity,max_seq_len=capacity,device=torch.device('cuda'),
       n_layers=len(model.model.layers),num_heads=model.config.num_key_value_heads,
       head_dim=model.config.hidden_size//model.config.num_attention_heads,
       disable_quant=model.cache_dtype=='float16',hadamard_dtype=None if model.cache_dtype=='float16' else torch.float16)

def clear(cache):
    cache.pages.zero_();cache.scales.zero_();cache.length=0
    cache._needs_init[:]=[True]*len(cache._needs_init)
    if hasattr(cache,'_metadata_length'):cache._metadata_length=None

def snapshot(cache):
    from .common import tensor_hash
    spec=cache.get_cache_specs_for_flash_infer(None)
    return {'length':cache.length,'needs_init':list(cache._needs_init),'pages_sha256':tensor_hash(cache.pages),
      'scales_sha256':tensor_hash(cache.scales),'metadata':{k:v.cpu().tolist() for k,v in spec.items() if k not in ('kv_data','kv_param')}}
