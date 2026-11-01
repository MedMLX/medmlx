# Third-party notices

Original MedMLX conversion code is licensed under [Apache-2.0](LICENSE).
The following upstream portions retain their attribution and terms.

## Apple Core AI Models

`recipes/sam3/export.py` adapts the full SAM 3 export wrapper from
[apple/coreai-models](https://github.com/apple/coreai-models), adding text
attention-mask inputs, fixed-grid position bias and independent source checks.
The upstream [BSD-3-Clause license](https://github.com/apple/coreai-models/blob/main/LICENSE)
is retained in full:

```text
Copyright 2026 Apple Inc.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## MedSAM2 and SAM 2

`recipes/medsam2/export.py` adapts encoder, prompt, mask, memory and rotary
attention operations from [MedSAM2](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383),
revision `332f30d420f1d1b08e2a79b3ae6a602458808383`, derived from
[SAM 2](https://github.com/facebookresearch/sam2). These portions retain the
MedSAM2 authors' and Meta Platforms, Inc. and affiliates' copyright and
Apache-2.0 terms; the full license is included in [LICENSE](LICENSE).
The recipe adds fixed export interfaces, real-valued rotary arithmetic,
memory masks and independent source comparisons.

## SynthSeg graph fixture

`tests/fixtures/synthseg-v1-graph.json` serializes the SynthSeg network
architecture from `ext/neuron/models.py` at
[BBillot/SynthSeg revision 2a2aa3bbfccb83f8253a51ca8b329b9938a2646d](https://github.com/BBillot/SynthSeg/tree/2a2aa3bbfccb83f8253a51ca8b329b9938a2646d).
The pinned repository has an Apache-2.0 root license, but that specific
[`ext/neuron/models.py`](https://github.com/BBillot/SynthSeg/blob/2a2aa3bbfccb83f8253a51ca8b329b9938a2646d/ext/neuron/models.py)
header explicitly states GPLv3 and requests citation of Dalca, Guttag and
Sabuncu, CVPR 2018. The root license does not resolve this file-specific
conflict. Redistribution clearance for the serialized graph remains unresolved;
confirm applicable terms with the publisher before releasing that fixture or
models derived from it. The numerical fixtures use synthetic inputs and are
not medical-image datasets or checkpoint weights.

## Dependencies and downloaded model materials

Source frameworks and converter packages are installed separately and retain
their own licenses. Recipes fetch source models and checkpoints separately;
those materials are not relicensed by this package. An exported model retains
its source and checkpoint terms, including any required notices and access or
use restrictions. Review the pinned model sources in [docs/survey.md](docs/survey.md)
before distributing an export.
