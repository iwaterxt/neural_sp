"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import operator
import torch

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler

class ChunkDataloader(DataLoader):

    def __init__(self, dataset, batch_size, distributed=False, num_workers=0, timeout=1000):
 
        if not distributed: 
            super(ChunkDataloader, self).__init__(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=self.collate_fn,
                                              timeout=timeout)
        else:
            import horovod.torch as hvd
            sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
            super(ChunkDataloader, self).__init__(dataset,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           num_workers=num_workers,
                                           collate_fn=self.collate_fn,
                                           drop_last=False,
                                           timeout=timeout)

    def collate_fn(self, batch):
        feats, utt_ids, labels = zip(*batch)
        data = {
            "utt_ids": utt_ids,
                "x": torch.FloatTensor(list(feats)),
                "y": torch.LongTensor(list(labels))
        }

        return data

      
class SeqDataloader(DataLoader):
    
    def __init__(self, dataset, batch_size, num_workers=0, distributed=False, test_only=False, timeout=1000):
        
        self.test_only = test_only
 
        # now decide on a sampler
        #base_sampler = torch.utils.data.SequentialSampler(self.dataset)
        base_sampler = torch.utils.data.RandomSampler(dataset)
        
        if not distributed:
            sampler = torch.utils.data.BatchSampler(base_sampler, batch_size, False)
            super(SeqDataloader, self).__init__(dataset,
                                           batch_sampler=sampler,
                                           num_workers=num_workers,
                                           collate_fn=self.collate_fn)
        else:
            import horovod.torch as hvd
            sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
            super(SeqDataloader, self).__init__(dataset,
                                           batch_size=batch_size, 
                                           sampler=sampler, 
                                           num_workers=num_workers, 
                                           collate_fn=self.collate_fn, 
                                           drop_last=False,
                                           timeout=timeout)
   

    def collate_fn(self, batch):

        xs,xlens,ys,ys_hist,ys_sub1,ys_sub2,utt_ids,speakers,sessions,text,feat_path,ys_prev,text_prev,ys_next,text_next=zip(*batch)
        data = {
            'xs': xs,
            'xlens': xlens,
            'ys': ys,
            'ys_hist': ys_hist,
            'ys_sub1': ys_sub1,
            'ys_sub2': ys_sub2,
            'utt_ids': utt_ids,
            'speakers': speakers,
            'sessions': sessions,
            'text': text,
            'feat_path': feat_path,  
            'ys_prev': ys_prev,
            'text_prev': text_prev,
            'ys_next': ys_next,
            'text_next': text_next,
        }
        
        return data 
