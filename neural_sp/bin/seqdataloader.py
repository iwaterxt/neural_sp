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
import horovod.torch as hvd
from torch.utils.data import Dataset, DataLoader
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.seq2seq.frontends.splicing import splice
from neural_sp.models.seq2seq.frontends.frame_stacking import stack_frame
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

        utt_ids = []
        x = []
        y = []

        print (type(batch))
        for item in batch:

        data = {
            "utt_ids": utt_ids,
                "x": torch.FloatTensor(x),
                "y": torch.LongTensor(y)
        }

        return data

      
class SeqDataloader(DataLoader):
    
    def __init__(self, dataset, batch_size, num_workers=0, distributed=False, 
                 num_stacks=1, num_splices=1, num_skips=1, pin_memory=False, test_only=False, timeout=1000):
        
        self.test_only = test_only
        self.num_stacks = num_stacks
        self.num_splices = num_splices
        self.num_skips = num_skips
        self.pin_memory = pin_memory
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
            sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
            super(SeqDataloader, self).__init__(dataset,
                                           batch_size=batch_size, 
                                           sampler=sampler, 
                                           num_workers=num_workers, 
                                           collate_fn=self.collate_fn,
                                           pin_memory=self.pin_memory,
                                           drop_last=False,
                                           timeout=timeout)
   

    def collate_fn(self, batch):

        xs = []
        xlens = []
        ys = []
        ys_hist = []
        ys_sub1 = []
        ys_sub2 = []
        utt_ids = []
        speakers = []
        sessions = []
        text = []
        for item in batch:
            xs.append(item['xs'][0])
            xlens.append(item['xlens'][0])
            ys.append(item['ys'][0])
            ys_hist.append(item['ys_hist'][0])
            ys_sub1.append(item['ys_sub1'])
            ys_sub2.append(item['ys_sub2'])
            utt_ids.append(item['utt_ids'][0])
            speakers.append(item['speakers'][0])
            sessions.append(item['sessions'][0])
            text.append(item['text'])

        if self.num_stacks > 1:
            xs = [stack_frame(x, self.num_stacks, self.num_skips)for x in xs]

        # Splicing
        if self.num_splices > 1:
            xs = [splice(x, self.num_splices, self.num_stacks) for x in xs]

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
            'text': text
        }
        
        return data 
