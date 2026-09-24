# Multi-Encoder Sphere Fusion for Complex-Scenario Speaker Verification

## Abstract

Complex speaker verification must preserve identity under child speech, same-gender interference, device and distance mismatch, reverberation, and sub-second utterances. We address these conditions with a multi-encoder system that combines complementary supervised and self-supervised representations. The model adds a wide residual speaker head to a frozen W2V-BERT 2.0 encoder, injects ReDimNet2 frame evidence through cross attention, and joins all branches with fixed unit-sphere fusion. The released system produces a 4,096-dimensional embedding and achieved **93.40746** on the semifinal public leaderboard of the “Voiceprint in the Mist” challenge.

## 1. Problem Formulation

For an utterance waveform $x$, the model extracts an embedding $e=f_\theta(x)$. Speaker similarity is computed only from the two embeddings:

$$
s(x_i,x_j)=\frac{e_i^\top e_j}{\lVert e_i\rVert_2\lVert e_j\rVert_2}.
$$

The inference graph consumes one waveform at a time. It does not use trial pairs, filenames, scenario labels, decision thresholds, or statistics collected from the evaluation set.

## 2. Architecture

### 2.1 Complementary acoustic views

The base representation combines ERes2Net, CAM++, ResNet293, ReDimNet2, and W2V-BERT 2.0. These encoders expose different inductive biases: convolutional speaker geometry, local temporal patterns, deep residual spectra, time-frequency frames, and contextual self-supervised states. We keep the pretrained encoders fixed during task training and learn compact heads on their outputs.

### 2.2 Wide depth-time residual head

Let $H_l$ be the hidden sequence from SSL layer $l$. Each layer passes through the released adapter $A_l$ and a zero-initialized wide residual adapter $R_l$:

$$
U_l=A_l(H_l)+R_l(H_l),\qquad
R_l(H_l)=W_{l,2}\operatorname{GELU}(W_{l,1}\operatorname{LN}(H_l)).
$$

The projected layer sequences are stacked along a depth axis. A depthwise two-dimensional convolution exchanges local evidence across depth and time before attentive statistics pooling. Zero initialization preserves the pretrained speaker geometry at the start of optimization and lets the residual learn only the correction required by the target domains.

### 2.3 Acoustic-to-SSL cross-encoder residual

ReDimNet2 provides an acoustic frame sequence $A$, while the wide SSL head provides $U$. Four-head cross attention treats SSL frames as queries and acoustic frames as keys and values:

$$
Q=W_Q\operatorname{LN}(U),\qquad [K,V]=W_{KV}\operatorname{LN}(A),
$$

$$
U'=U+W_O\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.
$$

The output projection $W_O$ is initialized to zero. The cross-encoder branch therefore begins as the wide residual model and learns a stable acoustic correction without disrupting the initial embedding space.

### 2.4 Unit-sphere fusion

Every branch is normalized before fusion. For the original GRL representation $g$ and cross-encoder representation $c$, the shared slot is

$$
z_{gc}=\operatorname{norm}\left(\operatorname{norm}(g)+\operatorname{norm}(c)\right).
$$

The final embedding concatenates a compressed geometry view, ReDimNet2, the contextual SSL view, $z_{gc}$, and the wide residual view with energy weights $0.20,0.40,0.15,0.15,0.10$, respectively:

$$
e=\big[\sqrt{0.20}\hat z_{geo};\sqrt{0.40}\hat z_{redim};
\sqrt{0.15}\hat z_{ssl};\sqrt{0.15}\hat z_{gc};
\sqrt{0.10}\hat z_{wide}\big].
$$

A fixed projection compresses the geometry block so the concatenation has 4,096 dimensions. All fusion coefficients are fixed at inference time.

## 3. Training

Task training uses five speech corpora and 2,077 speaker identities, including 317 child speakers. Each batch samples 0.5, 1, 2, and 3 second crops with probabilities 0.40, 0.35, 0.15, and 0.10. The pretrained encoders remain frozen; only the task heads and residual paths are optimized.

The objective is

$$
\mathcal L=0.30\mathcal L_{cls}+\mathcal L_{proto}+\mathcal L_{hard}
+0.75\mathcal L_{comp}+0.75\mathcal L_{dur}
+2.0\mathcal L_{coord}+\mathcal L_{aff}.
$$

Here, $\mathcal L_{cls}$ is AAM-Softmax classification, $\mathcal L_{proto}$ preserves speaker prototypes, $\mathcal L_{hard}$ separates nearest competing identities, $\mathcal L_{comp}$ reconstructs reliable identity evidence from short crops, $\mathcal L_{dur}$ enforces duration consistency, and the coordinate and affinity terms preserve the relational geometry of the teacher space. The released configuration uses seed 823, AAM scale 30 and margin 0.2, 20 warm-up steps, and 1,200 task-training steps.

## 4. Inference and Reproduction

Audio is converted to 16 kHz mono PCM before feature extraction. The runtime emits one finite `float32[4096]` embedding per WAV file; verification uses cosine similarity. Model assets are distributed separately because of repository size limits.

See the [Chinese README](../README.md) or [English README](../README_EN.md) for environment setup, weight installation, direct inference, and end-to-end task training commands.
