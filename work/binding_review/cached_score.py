"""Exact canonical-sequence scoring with shared-prefix KV caches."""
from collections import defaultdict
from contextlib import contextmanager

import torch
from transformers import DynamicCache


class CandidateScorer:
    def __init__(self, engine, candidates, batch_size=4, method="tree", precision="float32"):
        self.engine = engine
        self.candidates = list(candidates)
        self.batch_size = batch_size
        self.method = method
        self.precision = precision
        self.compute_dtype = torch.float32 if precision == "float32" else torch.bfloat16
        self.weight = engine.model.lm_head.weight.detach().to("cuda")
        self.tokens = [
            engine.tok.encode(str(value), add_special_tokens=False) + [engine.tok.eos_token_id]
            for value in self.candidates
        ]
        assert len({tuple(row) for row in self.tokens}) == len(self.tokens)
        self.prefixes = defaultdict(set)
        for tokens in self.tokens:
            for j, token in enumerate(tokens):
                self.prefixes[tuple(tokens[:j])].add(token)
        self.nodes = sorted((prefix for prefix in self.prefixes if prefix), key=lambda v: (len(v), v))
        self.node_index = {prefix: i+1 for i, prefix in enumerate(self.nodes)}
        self.node_index[()] = 0

    def logits(self, hidden):
        return torch.nn.functional.linear(hidden.float(), self.weight).log_softmax(-1)

    @torch.no_grad()
    def cache_score_group(self, sequences):
        model = self.engine.model.model
        ids = torch.tensor(sequences)
        embeddings = model.embed_tokens(ids).to("cuda", dtype=self.compute_dtype)
        output = model(inputs_embeds=embeddings, use_cache=True)
        cache = output.past_key_values
        logp = self.logits(output.last_hidden_state[:, -1])
        current = [()]
        current_scores = torch.zeros((len(sequences), 1), device="cuda")
        answers = {}
        for depth in range(max(map(len, self.tokens))):
            choices = [(i, prefix + (token,)) for i, prefix in enumerate(current)
                       for token in sorted(self.prefixes[prefix])]
            parents = torch.tensor([i for i, _ in choices], device="cuda")
            tokens = torch.tensor([prefix[-1] for _, prefix in choices], device="cuda")
            batch = len(sequences)
            indices = torch.arange(batch, device="cuda")[:, None] * len(current) + parents[None, :]
            scores = current_scores[:, parents] + logp[indices, tokens]
            continuing = []
            for j, (_, prefix) in enumerate(choices):
                if prefix in self.prefixes:
                    continuing.append(j)
                else:
                    answers[prefix] = scores[:, j].cpu()
            if not continuing:
                break
            select = torch.tensor(continuing, device="cuda")
            parent_indices = indices[:, select].reshape(-1)
            cache.batch_select_indices(parent_indices)
            next_ids = tokens[select].repeat(batch).cpu()[:, None]
            embeddings = model.embed_tokens(next_ids).to("cuda", dtype=self.compute_dtype)
            output = model(inputs_embeds=embeddings, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logp = self.logits(output.last_hidden_state[:, -1])
            current = [choices[j][1] for j in continuing]
            current_scores = scores[:, select]
            if torch.cuda.max_memory_reserved() > 6 * 1024**3:
                raise RuntimeError("6 GiB reserved-memory limit exceeded")
        return torch.stack([answers[tuple(tokens)] for tokens in self.tokens], dim=1)

    @torch.no_grad()
    def tree_score_group(self, sequences):
        model = self.engine.model.model
        prompt_length = len(sequences[0])
        suffix = [prefix[-1] for prefix in self.nodes]
        ids = torch.tensor([sequence+suffix for sequence in sequences])
        length = ids.shape[1]
        mask = torch.full((length, length), float("-inf"), device="cuda", dtype=self.compute_dtype)
        mask[:prompt_length, :prompt_length] = torch.triu(
            torch.full((prompt_length, prompt_length), float("-inf"), device="cuda",
                       dtype=self.compute_dtype), diagonal=1)
        for i, prefix in enumerate(self.nodes):
            row = prompt_length+i
            mask[row, :prompt_length] = 0
            for depth in range(1, len(prefix)+1):
                mask[row, prompt_length+self.node_index[prefix[:depth]]-1] = 0
        positions = list(range(prompt_length)) + [prompt_length+len(prefix)-1 for prefix in self.nodes]
        positions = torch.tensor([positions], device="cuda")
        embeddings = model.embed_tokens(ids).to("cuda", dtype=self.compute_dtype)
        hidden = model(inputs_embeds=embeddings, attention_mask=mask[None, None],
                       position_ids=positions, use_cache=False).last_hidden_state
        logp = self.logits(hidden[:, prompt_length-1:])
        scores = torch.zeros((len(sequences), len(self.tokens)), device="cuda")
        for j, tokens in enumerate(self.tokens):
            parents = [self.node_index[tuple(tokens[:i])] for i in range(len(tokens))]
            scores[:, j] = logp[:, parents, tokens].sum(-1)
        return scores.cpu()

    @contextmanager
    def evaluation_precision(self):
        frozen = [(parameter, parameter.dtype) for parameter in self.engine.model.model.parameters()
                  if parameter.device.type == "cuda" and not parameter.requires_grad]
        try:
            if self.precision == "float32":
                for parameter, _ in frozen:
                    parameter.data = parameter.data.float()
            yield
        finally:
            for parameter, dtype in frozen:
                parameter.data = parameter.data.to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, rows):
        with self.evaluation_precision():
            return self.score_rows(rows)

    @torch.no_grad()
    def gold(self, rows):
        output = []
        with self.evaluation_precision():
            for start in range(0, len(rows), 16):
                group = rows[start:start+16]
                sequences, positions, labels, sizes = [], [], [], []
                for i, row in enumerate(group):
                    prompt = self.engine.encode(row["text"])
                    target = self.engine.tok.encode(str(row["answer"]), add_special_tokens=False)
                    target += [self.engine.tok.eos_token_id]
                    sequences.append(prompt+target[:-1])
                    positions.extend((i,len(prompt)-1+j) for j in range(len(target)))
                    labels.extend(target)
                    sizes.append(len(target))
                length = max(map(len,sequences))
                ids = torch.tensor([s+[0]*(length-len(s)) for s in sequences])
                model = self.engine.model.model
                embeddings = model.embed_tokens(ids).to("cuda", dtype=self.compute_dtype)
                hidden = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
                batch, index = torch.tensor(positions, device="cuda").unbind(1)
                logp = self.logits(hidden[batch,index])
                selected = logp.gather(1,torch.tensor(labels,device="cuda")[:,None]).flatten()
                output.extend(float(values.sum()) for values in selected.split(sizes))
        return output

    def score_rows(self, rows):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            sequence = self.engine.encode(row["text"])
            groups[len(sequence)].append((i, sequence))
        output = torch.empty((len(rows), len(self.candidates)), dtype=torch.float64)
        for group in groups.values():
            for start in range(0, len(group), self.batch_size):
                chunk = group[start:start + self.batch_size]
                sequences = [sequence for _, sequence in chunk]
                scores = (self.tree_score_group(sequences) if self.method == "tree"
                          else self.cache_score_group(sequences))
                if torch.cuda.max_memory_reserved() > 6 * 1024**3:
                    raise RuntimeError("6 GiB reserved-memory limit exceeded")
                output[[i for i, _ in chunk]] = scores.double()
        return output.numpy()
