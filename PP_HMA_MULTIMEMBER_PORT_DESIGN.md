# PP>1 + Hybrid-KV (HMA) disagg: multi-member region recovery — port design & status

**Status: DESIGN / BLOCKED-ON-FOUNDATION. No connector behavior change in this commit.**

This branch is a working record for enabling NIXL KV transfer under
`pipeline_parallel_size > 1` with hybrid (Mamba + FullAttention) KV cache layouts
(HMA), the case currently guarded out in
`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`:

```python
# base_worker.py (current main)
if self.pp_size > 1 and self._is_hma_required:
    raise NotImplementedError(
        "NixlPushConnector does not support pipeline_parallel_size > 1 "
        "with hybrid KV cache layouts (HMA) yet."
    )
```

Tracking: vLLM issue #46407 (hybrid-model KV under PP disagg gives coherent-but-wrong
decode). Prior art on the push connector: #45880 (PP prefill in push mode).

## The bug this targets

Under PP>1 + HMA, the general-case allocator pools **one layer from every
`kv_cache_group`** into a single shared `KVCacheTensor` (same base address, offset 0).
For a multi-Mamba-group hybrid (e.g. Nemotron-3-Ultra: N Mamba groups + 1
FullAttention) each pooled tensor holds N Mamba layers + 1 FA layer. The NIXL
producer dedups by base address and advertises **only the group-0 representative**,
dropping the other N-1 Mamba members and the FA member. The consumer then never
receives descriptors for those layers, so the majority of Mamba layers decode on
stale/zero SSM state → fluent-but-wrong output. (PP1 is unaffected: each layer is its
own representative.)

## Why the existing fix does not port as a cherry-pick

The validated fix (`dmvevents/vllm@pr4/hma-multimember-region-recovery`, live byte-match
on Nemotron-3-Ultra 550B TP8/PP2, 2026-06-26) is built on a **per-layer-name HMA
routing foundation** (an unmerged stack: PP-aware handshake aggregation →
per-layer-name region routing → this recovery). That foundation branched from `main`
in **May 2026** and encodes HMA membership on the wire as
`registered_layer_names` + `region_members`, resolved on the consumer via
`_layer_name_to_kv_group_index` / `_local_layer_name_to_region_indices`.

Current `main` took a **different design** for the same problem and is ~2600 commits
ahead of that branch point:

- **PP** is handled by a *region-index offset* (`_remote_region_offset`,
  `base_worker.py` register path), not layer-name routing. It asserts a **uniform
  regions-per-layer** invariant (`num_regions % num_local_layers == 0`) — the exact
  invariant that re-advertising extra HMA members would break.
- **Descriptor selection** (`_compute_desc_ids`) is *group-based*: `block_ids` arrives
  as one list per `kv_cache_group`, and each group reads across `num_regions` (FA) or
  `num_ssm_regions` (SSM) with per-group strides. It uses `self._group_spec_types[i]`,
  never per-layer names.
- The connector was split `worker.py` → `base_worker.py` + `pull_worker.py` +
  `push_worker.py`. The old fix's single-file `worker.py` no longer exists.
- Wire metadata (`NixlAgentMetadata`, `NIXL_CONNECTOR_VERSION`) carries **none** of the
  fields the recovery reads (`registered_layer_names`, `region_members`, `pp_rank`, …);
  those came from the unmerged foundation.

None of the 11 symbols the recovery depends on exist on `main`:
`region_group_ids`, `registered_layer_names`, `region_members`,
`_layer_name_to_kv_group_index`, `_member_to_local_region`,
`_local_layer_name_to_region_indices`, `local_seen_layer_names`,
`get_backend_aware_kv_block_len`, `mamba_region_group_ids`, `_ssm_layer_names`,
`local_region_indices`.

So this is **not** a rebase/cherry-pick and **not** a ~600-line mechanical port. Adding
only the wire fields would be dead (nothing reads them) *and* interop-unsafe (msgspec
schema change). Porting only the producer re-advertise would violate the region
uniformity assertion and leave the consumer unable to receive the extra regions —
i.e. a broken branch.

## The correct path (re-express in main's idiom)

Two self-consistent designs, both requiring live 2-node byte-match to validate:

**Option A — group-based recovery (no new wire fields).** Investigate whether
`_compute_desc_ids`'s existing group-based path already emits descriptors for every
Mamba group's block table (it iterates all groups and reads `num_ssm_regions` across
regions). If the gap is only in *producer advertisement* under PP region-slicing, the
fix may reduce to correcting `_remote_region_offset` / region-slicing for the pooled
HMA case, keeping the group model intact. Smallest possible surface; verify first.

**Option B — port the layer-name foundation, then the recovery.** Re-implement the
`#43368`-equivalent per-layer-name HMA routing on top of the current push/pull split
(wire `region_members` on `NixlAgentMetadata` with a version bump, resolve members to
local regions on the consumer, re-advertise dedup'd cross-spec members in
`register_kv_caches`), then lift both PP>1+HMA guards. Larger surface; matches the
validated design exactly.

**Gate for either:** 2-node live run on a multi-Mamba-group hybrid, PP>1 prefill,
byte-identical decode vs the PP1 aggregated control (RDMA counter delta + coherent
`17x23=391`-class arithmetic check). GPU/capacity-gated.

Until one of these lands and passes the live gate, the `NotImplementedError` guards on
`base_worker.py` are correct and stay in place.
